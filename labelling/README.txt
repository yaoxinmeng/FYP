STEPS TO TAKE
*************
1. Run "python resize.py input_path --out_dir outputh_path"
   This changes image from 256 x 256 to 512 x 512 and renames files
   output_path should be named cleaned_data

2. Run "python label.py"
   This is for labelling the data from cleaned_data and saves output to labelled_data

3. Run "python augmentation.py labelled_data_folder"
   This is done after all data has been labelled and will generate 4x the original number of images, 
   as well as a .npz file

4. Run "python train_model.py"
   This trains the model based on files in the labelled_data directory

5. Upload the generated "outfile.npz" to dropbox/public
   Update the colab link if necessary

6. Run "python generate_sketch.py file_name" 
   This generates line equations from label files