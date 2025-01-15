import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import json
from tqdm import tqdm


def load_input_samples(data_dir, batch_size=32):
    """
    Load input image samples and preprocess them for Inception-v3.
    """
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def extract_layer_activations(model, dataloader, layer_name):
    """
    Extract activations from the specified layer for all input samples.
    """
    activations = []

    def hook(module, input, output):
        activations.append(output.detach().cpu())

    target_layer = dict(model.named_modules())[layer_name]
    hook_handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.cuda()
            model(inputs)

    hook_handle.remove()
    return torch.cat(activations, dim=0)


def load_concept_data(subject_path, subject_index_path, background_path, background_index_path):
    """
    Load concept activation vectors and their indices for subject and background.
    """
    subject_concepts = np.load(subject_path)
    background_concepts = np.load(background_path)

    with open(subject_index_path, 'r') as f:
        subject_indices = json.load(f)

    with open(background_index_path, 'r') as f:
        background_indices = json.load(f)

    return subject_concepts, subject_indices, background_concepts, background_indices


def modify_activations(activations, concepts, concept_indices):
    batch_size, channels, height, width = activations.shape

    num_concepts, flattened_size = concepts.shape  # (num_concepts, feature_dim)
    feature_dim = channels * height * width
    assert flattened_size == feature_dim, "Concept dimensions do not match activations"

    concepts_tensor = torch.tensor(concepts, dtype=torch.float32, device=activations.device)
    concepts_reshaped = concepts_tensor.view(num_concepts, channels, height,
                                             width)  # (num_concepts, channels, height, width)

    projections = torch.zeros_like(activations)  # Shape: (batch_size, channels, height, width)

    for i in concept_indices:
        concept_tensor = concepts_reshaped[i]  # Shape: (channels, height, width)

        concept_projection = (activations * concept_tensor).sum(dim=(1, 2, 3),
                                                                keepdim=True)  # Shape: (batch_size, 1, 1, 1)
        projections += concept_projection * concept_tensor

    activations_modified = activations - projections

    return activations_modified


def compute_model_output(model, modified_activations, layer_name):
    """
    Replace the activations of the specified layer with modified activations and compute the model output.
    """
    device = next(model.parameters()).device

    def hook(module, input, output):
        return modified_activations.to(device)

    target_layer = dict(model.named_modules())[layer_name]
    hook_handle = target_layer.register_forward_hook(hook)

    batch_size = modified_activations.shape[0]
    fake_input = torch.randn(batch_size, 3, 299, 299).to(device)

    with torch.no_grad():
        output = model(fake_input)

    hook_handle.remove()
    return output


def sigmoid(x):
    return 1 / (1 + torch.exp(-x)) if isinstance(x, torch.Tensor) else 1 / (1 + np.exp(-x))


# ReLU
def relu(x):
    return torch.maximum(x, torch.tensor(0.0)) if isinstance(x, torch.Tensor) else np.maximum(x, 0)


# Tanh
def tanh(x):
    return torch.tanh(x) if isinstance(x, torch.Tensor) else np.tanh(x)


# Softmax
def softmax(x, dim=0):
    return torch.softmax(x, dim=dim) if isinstance(x, torch.Tensor) else np.exp(x) / np.sum(np.exp(x), axis=dim)


def shapley_value_computation(concepts, activations, model, layer_name, target_class, num_samples):
    num_concepts = len(concepts)
    shapley_values = np.zeros(num_concepts)

    for _ in tqdm(range(num_samples)):
        permuted_indices = np.random.permutation(num_concepts)
        current_score = 0

        for i, concept_idx in enumerate(permuted_indices):
            subset_indices = permuted_indices[:i + 1]
            modified_activations = modify_activations(activations, concepts, subset_indices)

            model_with_modified = compute_model_output(model, modified_activations, layer_name)
            target_score = model_with_modified[:, target_class].mean().item()

            marginal_contribution = target_score - current_score

            shapley_values[concept_idx] += marginal_contribution / num_samples

            current_score = target_score

    shapley_values = softmax(shapley_values)

    return shapley_values


def save_shapley_results(shapley_values, subject_indices, background_indices, output_file):
    subject_results = [(subject_indices[i], shapley_values[i]) for i in range(len(subject_indices))]
    background_results = [(background_indices[i], shapley_values[len(subject_indices) + i]) for i in
                          range(len(background_indices))]

    all_results = subject_results + background_results

    all_results.sort(key=lambda x: x[1], reverse=True)

    with open(output_file, 'w') as f:
        for index, value in all_results:
            if index in subject_indices:
                f.write(f"{index} (Subject): {value:.4f}\n")
            else:
                f.write(f"{index} (Background): {value:.4f}\n")


def main(args):
    dataloader = load_input_samples(args.input_sample_dir, args.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.inception_v3(pretrained=True).eval().to(device)

    activations = extract_layer_activations(model, dataloader, args.layer_name)

    subject_concepts, subject_indices, background_concepts, background_indices = load_concept_data(
        args.subject_concepts_path, args.subject_indices_path, args.background_concepts_path,
        args.background_indices_path
    )

    all_concepts = np.vstack([subject_concepts, background_concepts])

    shapley_values = shapley_value_computation(all_concepts, activations, model, args.layer_name, args.target_class,
                                               args.num_samples)

    save_shapley_results(shapley_values, subject_indices, background_indices, args.output_file)

    print(f"Shapley values saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shapley Value Computation for Concept-based Model Explanation")

    parser.add_argument("--input_sample_dir", type=str, required=True,
                        help="Directory containing the input image samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing images")
    parser.add_argument("--layer_name", type=str, required=True, help="Layer name to extract activations from")
    parser.add_argument("--subject_concepts_path", type=str, required=True, help="Path to the subject concepts (npy)")
    parser.add_argument("--subject_indices_path", type=str, required=True, help="Path to the subject indices (json)")
    parser.add_argument("--background_concepts_path", type=str, required=True,
                        help="Path to the background concepts (npy)")
    parser.add_argument("--background_indices_path", type=str, required=True,
                        help="Path to the background indices (json)")
    parser.add_argument("--target_class", type=int, required=True,
                        help="Target class index for the Shapley value computation")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of samples to compute Shapley values")
    parser.add_argument("--output_file", type=str, required=True, help="File to save the Shapley values")

    args = parser.parse_args()

    main(args)
