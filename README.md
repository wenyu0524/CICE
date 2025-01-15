# Shapley Value Computation for Concept-based Model Explanation

This project computes Shapley values to explain the predictions of a deep learning model (Inception-v3) based on concept-based features (subject and background concepts). The Shapley values are used to evaluate the importance of each concept in influencing the model's decision for a given target class.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- tqdm
- numpy
- json

You can install the required dependencies using pip:

```bash
pip install torch torchvision tqdm numpy
```

Usage

To compute Shapley values for the model, you can use the following command:
```bash
python shapley_value_computation.py \
    --input_sample_dir <path_to_input_samples> \
    --layer_name <layer_name_to_extract_activations_from> \
    --subject_concepts_path <path_to_subject_concepts.npy> \
    --subject_indices_path <path_to_subject_indices.json> \
    --background_concepts_path <path_to_background_concepts.npy> \
    --background_indices_path <path_to_background_indices.json> \
    --target_class <target_class_index> \
    --num_samples <number_of_samples_for_computation> \
    --output_file <output_file_to_save_results>
Arguments
--input_sample_dir: Path to the directory containing input image samples.
--layer_name: The name of the layer to extract activations from (e.g., 'Mixed_7c').
--subject_concepts_path: Path to the file containing subject concepts (in .npy format).
--subject_indices_path: Path to the file containing subject concept indices (in .json format).
--background_concepts_path: Path to the file containing background concepts (in .npy format).
--background_indices_path: Path to the file containing background concept indices (in .json format).
--target_class: Index of the target class for which Shapley values will be computed.
--num_samples: Number of samples to use for Shapley value computation (default is 10).
--output_file: Path to the output file where Shapley values will be saved.
Output
The Shapley values for each concept (subject and background) are saved to the specified output_file. The values are sorted by their importance, with the highest Shapley value first.

Example Output
python
0 (Subject): 0.1234
1 (Subject): 0.1101
2 (Background): 0.0987
...
License
This project is licensed under the MIT License - see the LICENSE file for details.

This README provides basic usage and description of the code without being too complicated. You can add more specific details as needed.
