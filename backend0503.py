import sys
import os
import spectral
import numpy as np
import torch
import cv2
from spectral import imshow
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem
from hyperspectral_ui import Ui_Dialog  # Import the UI
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtCore import Qt, QRectF
from segment_anything import sam_model_registry, SamPredictor
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class BoundingBox(QGraphicsRectItem):
    """A resizable and movable bounding box for object selection."""
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.setPen(QPen(QColor(255, 0, 0), 2))  # Red border
        self.setBrush(QColor(255, 0, 0, 50))  # Semi-transparent red fill
        self.setFlag(QGraphicsRectItem.ItemIsMovable)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges)

class UI_Checker(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.current_folder = ""  
        self.image_path = None  
        self.hdr_path = None  
        self.hdr_data = None  
        self.current_band = 0  

        # Load folder icon
        self.folder_icon_path = "folder_icon.png"
        if not os.path.exists(self.folder_icon_path):
            self.folder_icon_path = os.path.join(os.path.dirname(__file__), "folder_icon.png")
        self.folder_icon = QtGui.QIcon(self.folder_icon_path) if os.path.exists(self.folder_icon_path) else QtGui.QIcon()

        # Disable user editing of textEdit
        self.textEdit.setReadOnly(True)

        # Connect UI elements to functions
        self.pushButton_2.clicked.connect(self.plot_spectral_signature)
        self.pushButton_3.clicked.connect(self.enable_drawing_mode)  # Connect SEGMENT INPUT button
        self.pushButton_4.clicked.connect(self.upload_folder)  
        self.pushButton_5.clicked.connect(self.navigate_back)
        self.pushButton_6.clicked.connect(self.analyze_segments)  # Connect ANALYZE SEGMENTS button
        self.pushButton_7.clicked.connect(self.clear_all_data)  # Corrected button name  
        self.listWidget.itemClicked.connect(self.on_folder_click)  
        self.horizontalSlider.valueChanged.connect(self.update_hdr_band)  
        self.pushButton.pressed.connect(self.show_png_image)  
        self.pushButton.released.connect(self.clear_display)  
        
        # Set up QGraphicsScene for image display
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        # Ensure doubleSpinBox updates with the slider
        self.doubleSpinBox.setDecimals(0)
        self.doubleSpinBox.setSingleStep(1)
        self.doubleSpinBox.valueChanged.connect(self.on_spinbox_value_changed)

        # Add bounding box properties
        self.drawing = False
        self.drawing_enabled = False
        self.bounding_box = None
        self.start_pos = None

        # Connect mouse events for the graphicsView
        self.graphicsView.mousePressEvent = self.start_drawing
        self.graphicsView.mouseMoveEvent = self.update_drawing
        self.graphicsView.mouseReleaseEvent = self.finish_drawing

        # Add storage for bounding box coordinates
        self.stored_box_coords = None

        # Add storage for current mask
        self.current_mask = None

        # Load SAM model (you can change the model type as needed)
        sam_model = sam_model_registry["vit_h"](checkpoint=r"C:\Users\ASUS\Desktop\sam\sam_vit_h_4b8939.pth")
        self.sam_predictor = SamPredictor(sam_model)

    def upload_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.current_folder = os.path.normpath(folder_path)  # Normalize path
            self.display_folder_contents(self.current_folder)

    def navigate_back(self):
        if self.current_folder:
            parent_folder = os.path.dirname(self.current_folder)
            if os.path.exists(parent_folder) and parent_folder != self.current_folder:
                self.current_folder = parent_folder
                self.display_folder_contents(parent_folder)

    def display_folder_contents(self, folder_path):
        """Displays the list of subfolders sorted by date (latest first)."""
        self.listWidget.clear()
        folder_path = os.path.normpath(folder_path)  

        if not os.path.exists(folder_path):
            print(f"Error: The folder '{folder_path}' does not exist.")
            return

        folders = [
            (item, os.path.getmtime(os.path.join(folder_path, item))) 
            for item in os.listdir(folder_path) 
            if os.path.isdir(os.path.join(folder_path, item))
        ]

        folders.sort(key=lambda x: x[1], reverse=True)  

        for folder_name, _ in folders:
            list_item = QListWidgetItem(self.folder_icon, folder_name)
            self.listWidget.addItem(list_item)

    def on_folder_click(self, item):
        """Handles clicking on a folder and processes it."""
        selected_folder = os.path.join(self.current_folder, item.text())
        selected_folder = os.path.normpath(selected_folder)  # Normalize path

        if os.path.exists(selected_folder):
            self.process_selected_folder(selected_folder)
        else:
            print(f"Error: The folder '{selected_folder}' does not exist.")

    def process_selected_folder(self, folder_path):
        """Clears previous data and processes the selected folder."""
        folder_path = os.path.normpath(folder_path)  

        if not os.path.exists(folder_path):
            print(f"Error: The folder '{folder_path}' does not exist.")
            return  

        self.clear_previous_data()
        self.current_folder = folder_path
        self.image_path = None  
        self.hdr_path = None  
        self.hdr_data = None  

        # Search for PNG in the selected folder
        for file in os.listdir(folder_path):
            if file.lower().endswith(".png"):
                self.image_path = os.path.join(folder_path, file)
                break  

        # Search for HDR in "capture" subfolder
        capture_folder_path = os.path.join(folder_path, "capture")
        capture_folder_path = os.path.normpath(capture_folder_path)

        if os.path.exists(capture_folder_path):
            for file in os.listdir(capture_folder_path):
                if file.endswith(".hdr"):
                    self.hdr_path = os.path.join(capture_folder_path, file)
                    break

        if self.hdr_path:
            self.load_hdr_file()

        self.update_ui()

    def clear_previous_data(self):
        """Clears all previously loaded data before loading a new folder."""
        self.scene.clear()  
        self.hdr_data = None
        self.image_path = None
        self.hdr_path = None
        self.current_band = 0
        self.horizontalSlider.setValue(0)
        self.doubleSpinBox.setValue(0)
        self.listWidget.clearSelection()  

    def update_ui(self):
        """Refresh the UI after processing a new folder."""
        self.scene.clear()  

        if self.hdr_data is not None:
            self.horizontalSlider.setMinimum(0)
            self.horizontalSlider.setMaximum(self.hdr_data.shape[2] - 1)
            self.doubleSpinBox.setMinimum(0)
            self.doubleSpinBox.setMaximum(self.hdr_data.shape[2] - 1)
            self.current_band = 0
            self.update_hdr_band()

        elif self.image_path and os.path.exists(self.image_path):
            self.show_png_image()  

    def load_hdr_file(self):
        try:
            hdr_image = spectral.open_image(self.hdr_path)
            self.hdr_data = hdr_image.load()

            if self.hdr_data is None or len(self.hdr_data.shape) < 3:
                raise ValueError("Invalid HDR file structure")

            # print(f"HDR data loaded. Shape: {self.hdr_data.shape}")  # Add this to check if data is loaded

            self.horizontalSlider.setMinimum(0)
            self.horizontalSlider.setMaximum(self.hdr_data.shape[2] - 1)
            self.doubleSpinBox.setMinimum(0)
            self.doubleSpinBox.setMaximum(self.hdr_data.shape[2] - 1)
            self.current_band = 0
            self.update_hdr_band()
        except Exception as e:
            print(f"Error loading HDR file: {e}")

    def update_hdr_band(self):
        if self.hdr_data is not None:
            self.current_band = self.horizontalSlider.value()
            self.doubleSpinBox.blockSignals(True)
            self.doubleSpinBox.setValue(self.current_band)
            self.doubleSpinBox.blockSignals(False)

            band_image = self.hdr_data[:, :, self.current_band]
            band_image = np.squeeze(band_image)  

            band_image = np.rot90(band_image, k=-1)  

            min_val, max_val = np.min(band_image), np.max(band_image)
            if max_val > min_val:
                band_image = ((band_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                band_image = np.zeros_like(band_image, dtype=np.uint8)  

            height, width = band_image.shape
            band_image_bytes = band_image.tobytes()
            image = QtGui.QImage(band_image_bytes, width, height, width, QtGui.QImage.Format_Grayscale8)
            pixmap = QtGui.QPixmap.fromImage(image)

            self.scene.clear()
            self.scene.addItem(QGraphicsPixmapItem(pixmap))
            self.graphicsView.fitInView(self.scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

    def on_spinbox_value_changed(self):
        new_value = int(self.doubleSpinBox.value())
        self.horizontalSlider.blockSignals(True)  
        self.horizontalSlider.setValue(new_value)
        self.horizontalSlider.blockSignals(False)  
        self.update_hdr_band()

    def show_png_image(self):
        if self.image_path and os.path.exists(self.image_path):
            pixmap = QtGui.QPixmap(self.image_path)
            self.scene.clear()
            self.scene.addItem(QGraphicsPixmapItem(pixmap))
            self.graphicsView.fitInView(self.scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)


    def restore_hdr_image(self):
        """Restore HDR display when the RGB Display button is released."""
        self.scene.clear()
        if self.hdr_data is not None:
            self.update_hdr_band()

    def enable_drawing_mode(self):
        """Enable drawing mode when SEGMENT INPUT is clicked"""
        self.drawing_enabled = True
        try:
            if self.bounding_box and self.bounding_box.scene():
                self.scene.removeItem(self.bounding_box)
        except RuntimeError:
            pass  # Handle case where object was already deleted
        self.bounding_box = None
        print("Click and drag on the image to draw a bounding box")

    def start_drawing(self, event):
        """Start drawing the bounding box"""
        if hasattr(self, 'drawing_enabled') and self.drawing_enabled and event.button() == Qt.LeftButton:
            self.drawing = True
            scene_pos = self.graphicsView.mapToScene(event.pos())
            self.start_pos = scene_pos

            # Remove existing bounding box if present
            if self.bounding_box:
                self.scene.removeItem(self.bounding_box)

            # Create new bounding box
            self.bounding_box = BoundingBox(
                scene_pos.x(),
                scene_pos.y(),
                0,  # Initial width
                0   # Initial height
            )
            self.scene.addItem(self.bounding_box)

    def update_drawing(self, event):
        """Update the bounding box size while drawing"""
        if self.drawing and self.bounding_box and self.start_pos:
            current_pos = self.graphicsView.mapToScene(event.pos())

            # Calculate width and height
            width = current_pos.x() - self.start_pos.x()
            height = current_pos.y() - self.start_pos.y()

            # Update bounding box geometry
            x = self.start_pos.x() if width >= 0 else current_pos.x()
            y = self.start_pos.y() if height >= 0 else current_pos.y()
            width = abs(width)
            height = abs(height)

            self.bounding_box.setRect(x, y, width, height)

    def finish_drawing(self, event):
        """Finish drawing the bounding box"""
        if self.drawing:
            self.drawing = False
            self.drawing_enabled = False  # Disable drawing mode after finishing
            if self.bounding_box:
                rect = self.bounding_box.rect()
                # Store the coordinates when finishing the drawing
                self.stored_box_coords = {
                    'x': rect.x(),
                    'y': rect.y(),
                    'width': rect.width(),
                    'height': rect.height()
                }
                print(f"Bounding box created: x={rect.x():.1f}, y={rect.y():.1f}, "
                      f"width={rect.width():.1f}, height={rect.height():.1f}")     

    def analyze_segments(self):
        # Ensure a bounding box exists
        if not self.stored_box_coords or self.image_path is None:
            print("No bounding box or image found for segmentation.")
            return

        # Load the image with OpenCV
        image = cv2.imread(self.image_path)
        if image is None:
            print(f"Error loading image: {self.image_path}")
            return

        # Prepare the image for SAM
        self.sam_predictor.set_image(image)

        # Get bounding box coordinates
        box_coords = self.stored_box_coords
        input_box = np.array([
            [box_coords['x'], box_coords['y']],  # top-left corner
            [box_coords['x'] + box_coords['width'], box_coords['y'] + box_coords['height']]  # bottom-right corner
        ])

        # Perform prediction
        masks, _, _ = self.sam_predictor.predict(box=input_box, multimask_output=False)

        # Store the first mask in self.current_mask
        self.current_mask = masks[0]

        # Convert the mask to a format suitable for PyQt and display it
        self.display_mask(self.current_mask, image.shape[:2])


    def display_mask(self, mask, image_shape):
        """Converts the binary mask to QImage and displays it on graphicsView."""
        # Ensure the mask is 8-bit (0-255) grayscale image
        mask = (mask * 255).astype(np.uint8)

        # Convert mask to QImage
        height, width = image_shape
        q_mask = QtGui.QImage(mask.data, width, height, width, QtGui.QImage.Format_Grayscale8)

        # Create QPixmap from the QImage
        pixmap = QtGui.QPixmap.fromImage(q_mask)

        # Clear the current scene and display the new mask
        self.scene.clear()
        self.scene.addItem(QGraphicsPixmapItem(pixmap))
        self.graphicsView.fitInView(self.scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

    def display_spectral_signature(self, spectral_signature):
        """Displays the spectral signature in graphicsView_2."""
        # Clear the previous plot
        if self.graphicsView_2.scene() is None:
            self.graphicsView_2.setScene(QGraphicsScene())  # Ensure a scene exists
        self.graphicsView_2.scene().clear()  # Clear all items in the scene
        
        # Create a new matplotlib figure
        figure = Figure(figsize=(5, 4))
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot(111)

        # Plot the average spectral signature
        ax.plot(spectral_signature, label="Average Spectral Signature")
        ax.set_xlabel("Band")
        ax.set_ylabel("Reflectance")
        ax.set_title("Spectral Signature of Segmented Region")
        ax.legend()

        # Add the matplotlib canvas to the QGraphicsScene
        scene = self.graphicsView_2.scene()  # Get the current scene
        scene.addWidget(canvas)  # Add the canvas as a widget to the scene

        # Update the graphics view with the new scene
        self.graphicsView_2.setScene(scene)
        
        # Save the reference to the canvas if needed (but no need to manually remove it)
        self.canvas = canvas


    def plot_spectral_signature(self):
        """Plots the average spectral signature of the segmented region."""
        if self.current_mask is None or self.hdr_data is None:
            print("Error: No mask or hyperspectral data available for plotting.")
            return

        mask = self.current_mask
        if mask.shape != self.hdr_data.shape[:2]:
            print(f"Error: Mask shape {mask.shape} does not match HDR data shape {self.hdr_data.shape[:2]}.")
            return

        # Apply mask to hyperspectral data across all bands
        masked_data = self.hdr_data[mask, :]

        if masked_data.size == 0:
            print("Error: The segmented region does not contain valid data.")
            return

        # Compute average spectral signature
        avg_spectral_signature = masked_data.mean(axis=0)

        # Display the plot
        self.display_spectral_signature(avg_spectral_signature)

        # Print statistics
        print(f"Plotted spectral signature across all bands")
        print(f"Min intensity: {np.min(avg_spectral_signature):.2f}")
        print(f"Max intensity: {np.max(avg_spectral_signature):.2f}")
        print(f"Mean intensity: {np.mean(avg_spectral_signature):.2f}")
       

    def clear_display(self):
        self.scene.clear()
        self.update_ui()

    def clear_all_data(self):
        """Clears all displayed data including images, plots, and masks."""
        # Clear the main image display
        self.scene.clear()

        # Clear the plot in graphicsView_2
        if hasattr(self, 'graphicsView_2'):
            if self.graphicsView_2.scene() is None:
                self.graphicsView_2.setScene(QGraphicsScene())
            self.graphicsView_2.scene().clear()

        # Reset all data
        self.hdr_data = None
        self.image_path = None
        self.hdr_path = None
        self.current_band = 0
        self.current_mask = None
        self.stored_box_coords = None

        # Reset UI elements
        self.horizontalSlider.setValue(0)
        self.doubleSpinBox.setValue(0)

        # Safely clear bounding box
        try:
            if self.bounding_box is not None:
                if self.bounding_box.scene() is not None:
                    self.scene.removeItem(self.bounding_box)
                self.bounding_box = None
        except RuntimeError:
            # Handle case where object was already deleted
            self.bounding_box = None

        # Preserve the original folder list without navigating into any subfolder
        if self.current_folder and os.path.exists(self.current_folder):
            self.display_folder_contents(self.current_folder)

        # Clear selection without changing content
        self.listWidget.clearSelection()

        print("All data cleared successfully")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = UI_Checker()
    window.show()
    sys.exit(app.exec_())
