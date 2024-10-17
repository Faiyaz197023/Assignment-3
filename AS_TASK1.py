import customtkinter as ctk  # CustomTkinter for modern GUI elements
from tkinter import filedialog, messagebox  # For file dialogs and message boxes
import numpy as np  # For handling arrays and image data
from PIL import Image, ImageTk  # For converting images to Tkinter-compatible format
from tf_keras.applications import ResNet50  # Pre-trained image classification model
from tf_keras.applications.resnet50 import preprocess_input, decode_predictions
from tf_keras.preprocessing import image
import re  # Regular expressions for file validation

"""
USING CUSTOMTKINTER: A MODERN LOOKING VERSION OF TKINTER FOR MORE AESTHETIC APPEARANCE.
"""

# Base class demonstrating Encapsulation and Abstraction principles
class AIModel:
    """
    Encapsulation: This class encapsulates AI model functionality, hiding implementation details from the user.
    Abstraction: It provides an abstract method that must be implemented by child classes, enforcing a common interface.
    """
    def __init__(self):
        pass

    def run_model(self, input_data):
        # Abstract method that must be overridden by child classes (Method Overriding)
        raise NotImplementedError("Subclasses should implement this method.")

# Child class for image classification demonstrating Inheritance and Method Overriding
class ImageClassificationModel(AIModel):
    """
    Inheritance: This class inherits from AIModel, which means it reuses its structure while modifying behavior.
    Method Overriding: It overrides the abstract run_model() method to provide functionality specific to image classification.
    Encapsulation: It hides the complexities of loading and using the ResNet50 model, only exposing an easy-to-use interface.
    """
    def __init__(self):
        super().__init__()  # Calls the parent constructor
        try:
            # Model is encapsulated and hidden from the outside world.
            self.model = ResNet50(weights='imagenet')  # Pre-trained ResNet50 model
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Error loading ResNet50 model: {str(e)}")

    def run_model(self, image_to_check):
        try:
            # Encapsulation: Image preprocessing and prediction logic are hidden inside this method.
            img = image.load_img(image_to_check, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Making predictions and decoding the results
            preds = self.model.predict(x)
            decoded_predictions = decode_predictions(preds, top=3)[0]

            # Returns results in a readable format
            return "\n".join([f"{pred[1]}: {pred[2] * 100:.2f}%" for pred in decoded_predictions])
        except FileNotFoundError:
            return "Error: File not found. Please select a valid image file."
        except Exception as e:
            return f"Error in Image Classification: {str(e)}"


# CustomTkinter Application Class demonstrating Encapsulation of GUI logic
class Application(ctk.CTk):
    """
    Encapsulation: This class encapsulates all the GUI logic and ensures all UI components are hidden and managed inside.
    Inheritance: Inherits from CTk (CustomTkinter's main class), allowing us to build the UI on top of a pre-existing framework.
    Polymorphism: Uses the same `run_model()` method for different models (polymorphism concept in action).
    """
    def __init__(self):
        super().__init__()  # Calls the parent class constructor (CTk)
        self.title("Modern AI Image Classifier")
        self.geometry("500x600")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # GUI Elements with enhanced aesthetics
        self.label = ctk.CTkLabel(self, text="Choose an image for classification:", font=("Arial", 18))
        self.label.pack(pady=20)

        # Button for selecting an image with custom appearance
        self.select_button = ctk.CTkButton(self, text="Select Image", command=self.select_image, corner_radius=8)
        self.select_button.pack(pady=10)

        # Button to run the classification with a sleek, modern look
        self.image_classification_button = ctk.CTkButton(self, text="Run Image Classification", command=self.run_image_classification, corner_radius=8)
        self.image_classification_button.pack(pady=10)

        # Label and Textbox for showing classification results with a refined layout
        self.result_label = ctk.CTkLabel(self, text="Classification Result:", font=("Arial", 16))
        self.result_label.pack(pady=20)

        self.result_text = ctk.CTkTextbox(self, height=200, width=400)
        self.result_text.pack(pady=10)

        # Variable to store the selected image path
        self.image_path = None

    def select_image(self):
        """
        Encapsulation: The image selection logic is encapsulated within this method, which hides the file handling details from the rest of the app.
        """
        try:
            self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
            if self.image_path:
                # Validate the file format using regular expressions
                if not re.match(r".*\.(jpg|jpeg|png)$", self.image_path, re.IGNORECASE):
                    raise ValueError("Selected file is not a valid image.")
                # Display success message
                messagebox.showinfo("Success", "Image successfully uploaded!")
        except ValueError as e:
            messagebox.showerror("Invalid File", str(e))
        except Exception as e:
            messagebox.showerror("File Selection Error", f"Error selecting file: {str(e)}")

    def run_image_classification(self):
        """
        Polymorphism: The run_model() method is used to perform image classification with different models (if needed).
        """
        try:
            if self.image_path:
                model = ImageClassificationModel()  # Using the ImageClassificationModel class
                result = model.run_model(self.image_path)
                # Display the result in the text widget
                self.result_text.delete(1.0, "end")
                self.result_text.insert("end", result)  # Insert new result
            else:
                messagebox.showerror("Error", "Please select an image first!")
        except Exception as e:
            messagebox.showerror("Image Classification Error", f"Error: {str(e)}")


# Main entry point for the application
if __name__ == "__main__":
    app = Application()  # Encapsulation: All functionality is wrapped in the Application class.
    app.mainloop()  # Starts the Tkinter main loop
