import torch
import tqdm
import pandas as pd
import os

def predict(model, loader, device):
   model.eval()  # Switch the model to evaluation mode
   id_to_pred = {}  # Save prediction results in the form: {image_id: predicted_label}

   with torch.no_grad():  # No gradient is needed during testing
      for images, img_paths in tqdm(loader, desc="Predicting", leave=False):
         # Move one batch of images to GPU or CPU
         images = images.to(device, non_blocking=True)

         # Forward pass: the model outputs classification scores for each class
         outputs = model(images)

         # Take the class with the highest score as the final prediction
         preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()

         for pred, path in zip(preds, img_paths):
            # Keep only the filename, for example:
            # /path/to/test_images/000001.jpg -> 000001.jpg
            image_id = os.path.basename(path)

            # Save prediction result for this image
            id_to_pred[image_id] = int(pred)

   return id_to_pred


def make_submission(id_to_pred, template_csv_path, output_path):
    # Read the official submission template file
    # This file usually contains at least the id column,
    # and may already contain a placeholder label column
    ids_df = pd.read_csv(template_csv_path)

    # Make sure the csv file really contains the id column
    if "id" not in ids_df.columns:
          raise ValueError(f"{template_csv_path} must contain 'id' column")
 
    # Copy the id list first, then create a new label column
    submission_df = ids_df.copy()
 
    # Fill each id with its predicted label using the dictionary
    submission_df["label"] = submission_df["id"].map(id_to_pred)
 
    # If any label is missing, it means some test images were not predicted correctly
    if submission_df["label"].isna().any():
       missing_ids = submission_df.loc[
          submission_df["label"].isna(), "id"
       ].head(5).tolist()
       raise RuntimeError(f"Missing predictions for some ids, examples: {missing_ids}")
 
    # Convert labels to integer format such as 0 or 1
    submission_df["label"] = submission_df["label"].astype(int)
 
    # Save the final submission file
    submission_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Step 1: get prediction results from the test set
    # TODO: fill the variable below
    model = ...
    test_loader = ...
    device = ...

    id_to_pred = predict(model, test_loader, device)

    # Step 2: choose where to save the generated csv file
    OUTPUT_DIR = "logs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")

    # Step 3: generate the final submission.csv
    # 获取当前脚本所在的目录
    current_path = os.path.dirname(os.path.abspath(__file__))

    # 获取当前脚本所在的项目根目录
    root_path = os.path.dirname(current_path)
    
    TEST_DATA_DIR = os.path.join(root_path, "data/test_images")
    
    make_submission(
       id_to_pred=id_to_pred,  # prediction dictionary returned by predict()
       template_csv_path=os.path.join(TEST_DATA_DIR, "sample_submission.csv"),  # official submission template
       output_path=submission_path,  # final csv save path
    )

    print(f"Kaggle submission file saved to {submission_path}")