import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Combobox
import os
import pickle
import pandas as pd
import numpy as np

from feature_extraction import extract_features

# load the scaler and the models
scaler = pickle.load(open('pickles/scaler.pkl', 'rb'))

model_files = {
    'Logistic Regression': 'pickles/logistic_regression_model.pkl',
    'Decision Tree': 'pickles/decision_tree_model.pkl',
    'Random Forest': 'pickles/random_forest_model.pkl'
}
models = {}
for model_name, model_file in model_files.items():
    model = pickle.load(open(model_file, 'rb'))
    models[model_name] = model

def classify_file():
    # get file path chosen by the user
    file_path = filedialog.askopenfilename()
    label_result.config(text="Analysing the uploaded file...")
    label_result.update_idletasks()

    # validate the file format
    valid_extensions = ['.mp3', '.wav']
    _, file_extension = os.path.splitext(file_path)
    if file_extension not in valid_extensions:
        label_result.config(text="Invalid file type")
        return

    # extract features from the music file
    data = extract_features(file_path)
    print(data)

    # standardise the data
    scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    print(scaled_data)

    # predict the genre using the model chosen by the user
    selected_model = model_var.get()
    if selected_model in models:
        model = models[selected_model]
        predicted_labels = model.predict(scaled_data)
    else:
        predicted_labels = []
    print(predicted_labels)

    # pick the majority vote from 5 labels
    unique_labels, label_counts = np.unique(predicted_labels, return_counts=True)
    max_count_index = np.argmax(label_counts)
    majority_label = unique_labels[max_count_index]
    print(majority_label)

    # display the predicted majority vote label
    label_result.config(text=f"Predicted genre: {majority_label}")

# main window
window = tk.Tk()
window.title("Music Genre Classification App")
window.geometry("400x200")
window.resizable(False, False)

# app description label
label_description = tk.Label(window, text="Music genre classification app, please upload an .mp3 or .wav file.")
label_description.grid(row=0, column=0, padx=10, pady=10)

# dropdown menu with models to choose
model_var = tk.StringVar()
model_var.set('Logistic Regression')

label_model = tk.Label(window, text="Choose Model:")
label_model.grid(row=1, column=0, padx=10)

dropdown_model = tk.OptionMenu(window, model_var, *models.keys())
dropdown_model.grid(row=2, column=0, padx=10)

# file upload button
upload_button = tk.Button(window, text="Upload File", command=classify_file)
upload_button.grid(row=3, column=0, padx=10, pady=20)

# prediction label
label_result = tk.Label(window, text="")
label_result.grid(row=4, column=0, padx=10)

window.mainloop()