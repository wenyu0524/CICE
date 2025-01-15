# Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- scikit-image
- matplotlib

You can install the required dependencies using pip:

```bash
pip install torch torchvision tqdm numpy
```


# Concepts Extraction

This project performs model interpretation by discovering concepts related to the target class using images from a specified directory. Statistical testing is applied to filter concepts, and random experiments are used for evaluation. The script uses a PyTorch model (e.g., Inception-v3) to analyze image features and concepts.

An example run command:

```bash
python ace_run.py --num_parallel_runs 0 --target_class timber_wolf --source_dir SOURCE_DIR_foreground --working_dir SAVE_DIR --model_to_run inception_v3 --labels_path imagenet_class_index.json --feature_names Mixed_7c --num_random_exp 20 --max_imgs 50 --min_imgs 30
```
where：
```bash
num_random_exp: number of random concepts with respect to which concept-activaion-vectors
```
For example if you set num_random_exp=20, you need to create folders random500_0, rando500_1, ..., random_500_19 and put them in the SOURCE_DIR where each folder contains a set of 50-500 randomly selected images of the dataset (ImageNet in our case).

SOURCE_DIR: Directory where the discovery images (refer to the paper) are saved. 

```bash
target_class: Name of the class which prediction is to be explained.
num_parallel_runs: Number of parallel jobs (loading images, etc). If 0, parallel processing is deactivated.
SAVE_DIR: Where the experiment results (both text report and the discovered concept examples) are saved.
model_to_run: Any torch.hub model. Note that you may need to edit the _get_gradients function in CICE.py.
```

# Shapley Value Computation

This project computes Shapley values to explain the predictions of a deep learning model (Inception-v3) based on concept-based features (subject and background concepts). The Shapley values are used to evaluate the importance of each concept in influencing the model's decision for a given target class.

## Usage

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
```
## Arguments
```bash
--input_sample_dir: Path to the directory containing input image samples.
--layer_name: The name of the layer to extract activations from (e.g., 'Mixed_7c').
--subject_concepts_path: Path to the file containing subject concepts (in .npy format).
--subject_indices_path: Path to the file containing subject concept indices (in .json format).
--background_concepts_path: Path to the file containing background concepts (in .npy format).
--background_indices_path: Path to the file containing background concept indices (in .json format).
--target_class: Index of the target class for which Shapley values will be computed.
--num_samples: Number of samples to use for Shapley value computation .
--output_file: Path to the output file where Shapley values will be saved.
```
## Output
The Shapley values for each concept (subject and background) are saved to the specified output_file. The values are sorted by their importance, with the highest Shapley value first.

## Example Output
```bash
0 (Subject): 0.1234
1 (Subject): 0.1101
2 (Background): 0.0987
...
```
## License
This project is licensed under the MIT License - see the LICENSE file for details.
