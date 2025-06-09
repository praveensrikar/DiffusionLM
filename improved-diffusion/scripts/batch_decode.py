import os, sys, glob

# Get the directory of the currently running script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Get the project root directory (which is one level up from 'scripts')
project_root = os.path.dirname(script_dir)

full_lst = glob.glob(sys.argv[1])
top_p = -1.0 if len(sys.argv) < 2 else sys.argv[2]
print(f'top_p = {top_p}')
pattern_ = 'model' if len(sys.argv) < 3 else sys.argv[3]
print(f'pattern_ = {pattern_}', sys.argv[3])

output_lst = []
for lst in full_lst:
    print(lst)
    try:
        # Find the model checkpoint file.
        tgt = sorted(glob.glob(f"{lst}/{pattern_}*pt"))[-1]
        lst = os.path.split(lst)[1]
        print(lst)
        num = 1
    except:
        continue
        
    model_arch = 'conv-unet' if 'conv-unet' in lst else 'transformer'
    mode = 'image' if ('conv' in model_arch) else 'text'
    print(mode)
    
    # Extract details from the filename to pass to the sampling script
    if 'e2e-tgt' in lst:
        modality = 'e2e-tgt'
    # Add other modality checks here if needed...
    else:
        modality = 'text' # Default modality

    out_dir = 'generation_outputs'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    num_samples = 50
    if modality == 'e2e':
        num_samples = 547

    # *** CORRECTED COMMAND ***
    # Construct the full path to the sampling script
    sample_script_path = os.path.join(script_dir, f'{mode}_sample.py')

    # Set the PYTHONPATH environment variable for the subprocess
    env_vars = f'PYTHONPATH=$PYTHONPATH:{project_root}'

    COMMAND = (
        f'{env_vars} python {sample_script_path} '
        f'--model_path {tgt} '
        f'--batch_size 50 '
        f'--num_samples {num_samples} '
        f'--top_p {top_p} '
        f'--out_dir {out_dir}'
    )
    print(f"Executing command: {COMMAND}")

    # Define the expected output path
    model_base_name = os.path.basename(os.path.split(tgt)[0]) + f'.{os.path.split(tgt)[1]}'
    if modality in ['e2e-tgt', 'e2e']:
        out_path2 = os.path.join(out_dir, f"{model_base_name}.samples_{top_p}.json")
    else:
        out_path2 = os.path.join(out_dir, f"{model_base_name}.samples_{top_p}.txt")

    # Check if the output already exists
    if not os.path.exists(out_path2):
        # Run the command and check its return code to ensure it was successful
        return_code = os.system(COMMAND)
        if return_code != 0:
            print(f"Error: The command '{COMMAND}' failed with return code {return_code}")
            sys.exit(return_code)

    # Verify the file exists before proceeding
    if not os.path.exists(out_path2):
        print(f"Error: Output file '{out_path2}' not found after running command.")
        sys.exit(1)

    output_lst.append(out_path2)
    print(f"Successfully generated: {out_path2}")

    # ... (rest of the evaluation script logic) ...
    # This part can be re-enabled once sample generation is successful.
    print('Sample generation complete. Skipping further evaluation for now.')
    continue


print('\nFinal output lists:')
print("\n".join(output_lst))