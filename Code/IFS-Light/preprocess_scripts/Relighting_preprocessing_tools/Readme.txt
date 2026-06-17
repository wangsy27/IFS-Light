Code for the paper: "IFS-Light: An Interactive Framework for Single-view Face Relighting with both Facial and Lighting Consistency".

------------------------------------------------------------------------------------------------
Please download the models and put everything into the Relighting_preprocessing_tools folder. For more model details, please refer to the folder "./Model".
The folder structure should be look like this:
.
└── Relighting_preprocessing_tools
    ├── Arcface                   <--- For face embeddings
    ├── DECA                      <--- For 3D render
    ├── diffae                    <--- For shadow
    ├── face-parsing.PyTorch      <--- For face segment
    ├── FFHQ_align                <--- Image alignment
    ├── skintone                  <--- For skin tone scale
    └── create_dataset.py	

Please running a preprocessing script:
Command: python create_dataset.py --image_dir <path_to_images> --out_dataset_dir <output_path> --ffhq_align --skintone --faceseg --albedo --arcface --shadow
Example: python create_dataset.py --image_dir /home/user/IFS-Light/preprocess_scripts/Relighting_preprocessing_tools/ffhq/aligned_images/valid/ --out_dataset_dir ./ffhq/ --skintone --faceseg --albedo --arcface --shadow