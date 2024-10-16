import tkinter as tk  # GUI framework
from tkinter import filedialog, messagebox  # Dialogs for file selection and error messages
import face_recognition  # Facial recognition library
import re  # Regular expressions for validating file types
from tf_keras.applications import ResNet50  # Pre-trained model for classification
from tf_keras.applications.resnet50 import preprocess_input, decode_predictions
from tf_keras.preprocessing import image
import numpy as np

# Base class demonstrating Encapsulation and Abstraction
class AIModel:
    # Encapsulation - Logic related to AI models is encapsulated in this class.
    # Abstraction - We are defining a generic interface that child classes will implement.
    def __init__(self):
        pass  # Constructor initializes the object, but internal details are abstracted.

    def run_model(self, input_data):
        # Abstract method to be overridden by subclasses (Method Overriding)
        pass

# Child class demonstrating Inheritance and Method Overriding
class FacialRecognitionModel(AIModel):
    # Inheritance - This class inherits from the AIModel, which means it gets access to all methods and properties of the parent class.
    def __init__(self):
        super().__init__()  # Calls the constructor of the parent class (AIModel).

    def run_model(self, image_to_check):
        # Method Overriding - This method overrides the abstract method in AIModel to provide a specific implementation for facial recognition.
        try:
            unknown_image = face_recognition.load_image_file(image_to_check)
            face_locations = face_recognition.face_locations(unknown_image)

            if len(face_locations) == 0:
                return "No face found"
            else:
                results = [
                    f"Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}"
                    for top, right, bottom, left in face_locations
                ]
                return "\n".join(results)
        except FileNotFoundError:
            return "Error: File not found. Please select a valid image file."
        except Exception as e:
            return f"Error in Facial Recognition: {str(e)}"

# Another child class demonstrating Inheritance and Polymorphism
class ImageClassificationModel(AIModel):
    # Inheritance - This class also inherits from AIModel, reusing the common structure.
    def __init__(self):
        super().__init__()  # Calls the parent class constructor.
        try:
            # Encapsulation - Model loading logic is encapsulated within this class.
            self.model = ResNet50(weights='imagenet')  # Load pre-trained model
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Error loading ResNet50 model: {str(e)}")

    def run_model(self, image_to_check):
        # Method Overriding - This method provides a specific implementation for image classification.
        try:
            img = image.load_img(image_to_check, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Polymorphism - Both models implement the `run_model()` method differently based on their purpose.
            preds = self.model.predict(x)
            decoded_predictions = decode_predictions(preds, top=3)[0]
            return "\n".join([f"{pred[1]}: {pred[2] * 100:.2f}%" for pred in decoded_predictions])
        except FileNotFoundError:
            return "Error: File not found. Please select a valid image file."
        except Exception as e:
            return f"Error in Image Classification: {str(e)}"

# Tkinter Application Class - Demonstrating Encapsulation of GUI logic
class Application(tk.Tk):
    # Multiple Inheritance - Inheriting from `tk.Tk` to create a GUI application.
    def __init__(self):
        super().__init__()  # Calls the constructor of Tk class (Inheritance from Tk).
        self.title("AI Model Integration App")

        # GUI elements encapsulated within the class (Encapsulation)
        self.label = tk.Label(self, text="Choose an image for processing:")
        self.label.pack(pady=10)

        self.select_button = tk.Button(self, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.facial_recognition_button = tk.Button(self, text="Run Facial Recognition", command=self.run_facial_recognition)
        self.facial_recognition_button.pack(pady=10)

        self.image_classification_button = tk.Button(self, text="Run Image Classification", command=self.run_image_classification)
        self.image_classification_button.pack(pady=10)

        self.text_output = tk.Text(self, height=10, width=50)
        self.text_output.pack(pady=10)

    def select_image(self):
        # Encapsulation - Image selection logic is hidden inside this method.
        try:
            self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
            if self.image_path:
                if not re.match(r".*\.(jpg|jpeg|png)$", self.image_path, re.IGNORECASE):
                    raise ValueError("Selected file is not a valid image.")
                self.text_output.insert(tk.END, f"Selected Image: {self.image_path}\n")
        except ValueError as e:
            messagebox.showerror("Invalid File", str(e))
        except Exception as e:
            messagebox.showerror("File Selection Error", f"Error selecting file: {str(e)}")

    def run_facial_recognition(self):
        # Polymorphism - The same method name `run_model()` behaves differently for each AI model.
        try:
            if hasattr(self, 'image_path'):
                model = FacialRecognitionModel()  # Creating an instance of FacialRecognitionModel
                result = model.run_model(self.image_path)  # Method Overriding in action
                self.text_output.insert(tk.END, f"Facial Recognition Result:\n{result}\n")
            else:
                messagebox.showerror("Error", "Please select an image first!")
        except Exception as e:
            messagebox.showerror("Facial Recognition Error", f"Error: {str(e)}")

    def run_image_classification(self):
        # Polymorphism - Running a different model with the same method call structure.
        try:
            if hasattr(self, 'image_path'):
                model = ImageClassificationModel()  # Creating an instance of ImageClassificationModel
                result = model.run_model(self.image_path)  # Method Overriding in action
                self.text_output.insert(tk.END, f"Image Classification Result:\n{result}\n")
            else:
                messagebox.showerror("Error", "Please select an image first!")
        except Exception as e:
            messagebox.showerror("Image Classification Error", f"Error: {str(e)}")

# Main entry point for the application
if __name__ == "__main__":
    app = Application()  # Multiple Inheritance from Tkinter's Tk class
    app.mainloop()  # Starts the GUI event loop

