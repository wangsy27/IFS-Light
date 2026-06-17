Code for the paper: "IFS-Light: An Interactive Framework for Single-view Face Relighting with both Facial and Lighting Consistency".

------------------------------------------------------------------------------------------------

To relight the image, you can run the following command:
Command:python relight.py --dataset <ffhq/mp/etc.> --set <train/valid> --step <ckpt_step> --out_dir <sampling_output> --cfg_name <cfg_name>.yaml --log_dir <ckpt_savename> --diffusion_steps <1000> --timestep_respacing <250/500/1000,etc> --sample_pair_json <path_to_sample_file> --sample_pair_mode pair --itp render_face --itp_step <number_of_frames> --batch_size <batch_size> --gpu_id <gpu_id> --lerp --idx <start_idx> <end_idx>

Example:python relight.py --dataset ffhqvalid --set valid --step 050000 --out_dir ./ffhq/ --cfg_name difareli_256.yaml --log_dir ours --diffusion_steps 1000 --timestep_respacing 250 --sample_pair_json ./ffhq.json --sample_pair_mode pair --itp render_face --itp_step 2 --batch_size 1 --gpu_id 0 --lerp --idx 0 10

The relighting results are in "./src=69954.png"

