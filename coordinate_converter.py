#!/usr/bin/env python3
"""
Coordinate system conversion for manually identified curves.
Converts pixel coordinates to chart coordinates (age vs height/weight).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


class CoordinateConverter:
    """Convert pixel coordinates to chart coordinates."""
    
    def __init__(self, image_path, debug=False):
        """Initialize the coordinate converter.
        
        Args:
            image_path (str): Path to the original chart image
            debug (bool): Whether to show debug information
        """
        self.image_path = image_path
        self.debug = debug
        self.image = None
        self.chart_bounds = None
        self.age_range = (2, 19)  # Age range from chart title
        self.height_range = None  # Will be determined from chart
        self.weight_range = None  # Will be determined from chart
        
        self.load_image()
        
    def load_image(self):
        """Load the chart image."""
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    def set_chart_bounds_interactive(self):
        """Interactively set the chart boundaries."""
        print("\n" + "="*60)
        print("CHART BOUNDARY SETUP")
        print("="*60)
        print("Click on the chart to define the coordinate system:")
        print("1. Click on the TOP-LEFT corner of the chart area")
        print("2. Click on the BOTTOM-RIGHT corner of the chart area")
        print("3. Close the window when done")
        print("="*60)
        
        self.boundary_points = []
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 20))
        ax.imshow(self.image)
        ax.set_title('Click: 1) Top-left corner, 2) Bottom-right corner of chart area')
        
        def on_click(event):
            if event.inaxes != ax:
                return
            
            x, y = int(event.xdata), int(event.ydata)
            self.boundary_points.append([x, y])
            
            # Draw the point
            ax.plot(x, y, 'ro', markersize=10)
            ax.text(x, y-50, f'Point {len(self.boundary_points)}', 
                   fontsize=12, color='red', fontweight='bold')
            
            print(f"Point {len(self.boundary_points)}: ({x}, {y})")
            
            if len(self.boundary_points) == 2:
                # Draw rectangle
                p1, p2 = self.boundary_points
                rect_x = [p1[0], p2[0], p2[0], p1[0], p1[0]]
                rect_y = [p1[1], p1[1], p2[1], p2[1], p1[1]]
                ax.plot(rect_x, rect_y, 'r-', linewidth=2)
                ax.set_title('Chart boundaries set! Close window to continue.')
                
            fig.canvas.draw()
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        
        if len(self.boundary_points) != 2:
            raise ValueError("Need exactly 2 points to define chart boundaries")
            
        # Set chart bounds
        p1, p2 = self.boundary_points
        self.chart_bounds = {
            'x_min': min(p1[0], p2[0]),
            'x_max': max(p1[0], p2[0]),
            'y_min': min(p1[1], p2[1]),
            'y_max': max(p1[1], p2[1])
        }
        
        print(f"Chart bounds set: {self.chart_bounds}")
        
    def set_value_ranges_interactive(self):
        """Interactively set the value ranges for the chart."""
        print("\nSetting value ranges for the chart...")
        print("Based on the chart title 'Weight-and-height Boys 2-19 years':")
        print(f"Age range: {self.age_range[0]} to {self.age_range[1]} years")
        
        # Get height range
        print("\nFor HEIGHT values:")
        height_min = input("Enter minimum height value (e.g., 80): ").strip()
        height_max = input("Enter maximum height value (e.g., 200): ").strip()
        
        try:
            self.height_range = (float(height_min), float(height_max))
        except ValueError:
            print("Using default height range: 80-200 cm")
            self.height_range = (80, 200)
            
        # Get weight range
        print("\nFor WEIGHT values:")
        weight_min = input("Enter minimum weight value (e.g., 10): ").strip()
        weight_max = input("Enter maximum weight value (e.g., 100): ").strip()
        
        try:
            self.weight_range = (float(weight_min), float(weight_max))
        except ValueError:
            print("Using default weight range: 10-100 kg")
            self.weight_range = (10, 100)
            
        print(f"Value ranges set:")
        print(f"  Age: {self.age_range}")
        print(f"  Height: {self.height_range}")
        print(f"  Weight: {self.weight_range}")
        
    def pixel_to_chart_coordinates(self, pixel_points, chart_type='height'):
        """Convert pixel coordinates to chart coordinates.
        
        Args:
            pixel_points (np.ndarray): Array of (x, y) pixel coordinates
            chart_type (str): 'height' or 'weight' to determine y-axis range
            
        Returns:
            np.ndarray: Array of (age, value) chart coordinates
        """
        if self.chart_bounds is None:
            raise ValueError("Chart bounds not set. Call set_chart_bounds_interactive() first.")
            
        chart_coords = []
        
        for x_pixel, y_pixel in pixel_points:
            # Convert x (pixel) to age
            age = self.age_range[0] + (x_pixel - self.chart_bounds['x_min']) * \
                  (self.age_range[1] - self.age_range[0]) / \
                  (self.chart_bounds['x_max'] - self.chart_bounds['x_min'])
            
            # Convert y (pixel) to value (note: y increases downward in images)
            if chart_type == 'height':
                value_range = self.height_range
            else:
                value_range = self.weight_range
                
            value = value_range[1] - (y_pixel - self.chart_bounds['y_min']) * \
                    (value_range[1] - value_range[0]) / \
                    (self.chart_bounds['y_max'] - self.chart_bounds['y_min'])
            
            chart_coords.append([age, value])
            
        return np.array(chart_coords)
    
    def process_manual_curves(self, curves_file, chart_type='height'):
        """Process manually identified curves and convert coordinates.
        
        Args:
            curves_file (str): Path to JSON file with manual curves
            chart_type (str): 'height' or 'weight'
            
        Returns:
            list: List of curves in chart coordinates
        """
        # Load manual curves
        if not Path(curves_file).exists():
            raise FileNotFoundError(f"Curves file not found: {curves_file}")
            
        with open(curves_file, 'r') as f:
            data = json.load(f)
            
        pixel_curves = data['curves']
        
        # Set up coordinate system if not done
        if self.chart_bounds is None:
            self.set_chart_bounds_interactive()
            self.set_value_ranges_interactive()
        
        # Convert each curve
        chart_curves = []
        for i, pixel_curve in enumerate(pixel_curves):
            pixel_points = np.array(pixel_curve)
            chart_points = self.pixel_to_chart_coordinates(pixel_points, chart_type)
            chart_curves.append(chart_points.tolist())
            
            if self.debug:
                print(f"Curve {i+1}: {len(pixel_points)} points converted")
        
        return chart_curves
    
    def save_processed_curves(self, chart_curves, output_file, chart_type='height'):
        """Save processed curves to file.
        
        Args:
            chart_curves (list): List of curves in chart coordinates
            output_file (str): Output file path
            chart_type (str): 'height' or 'weight'
        """
        data = {
            'chart_type': chart_type,
            'age_range': self.age_range,
            'value_range': self.height_range if chart_type == 'height' else self.weight_range,
            'chart_bounds_pixels': self.chart_bounds,
            'num_curves': len(chart_curves),
            'curves': chart_curves
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved {len(chart_curves)} processed curves to {output_file}")
        
        # Also save as CSV for easy use
        csv_file = output_file.replace('.json', '.csv')
        with open(csv_file, 'w') as f:
            f.write(f"curve_id,age,{chart_type}\n")
            for curve_id, curve in enumerate(chart_curves):
                for age, value in curve:
                    f.write(f"{curve_id+1},{age:.2f},{value:.2f}\n")
                    
        print(f"Also saved as CSV: {csv_file}")
    
    def visualize_processed_curves(self, chart_curves, chart_type='height'):
        """Visualize the processed curves in chart coordinates.
        
        Args:
            chart_curves (list): List of curves in chart coordinates
            chart_type (str): 'height' or 'weight'
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Show original image with pixel curves
        ax1.imshow(self.image)
        ax1.set_title('Original Image with Chart Boundaries')
        
        if self.chart_bounds:
            # Draw chart boundaries
            bounds = self.chart_bounds
            rect_x = [bounds['x_min'], bounds['x_max'], bounds['x_max'], bounds['x_min'], bounds['x_min']]
            rect_y = [bounds['y_min'], bounds['y_min'], bounds['y_max'], bounds['y_max'], bounds['y_min']]
            ax1.plot(rect_x, rect_y, 'r-', linewidth=3, label='Chart Boundaries')
            ax1.legend()
        
        ax1.axis('off')
        
        # Show processed curves
        colors = plt.cm.tab20(np.linspace(0, 1, len(chart_curves)))
        
        for i, curve in enumerate(chart_curves):
            curve_array = np.array(curve)
            ax2.plot(curve_array[:, 0], curve_array[:, 1], 
                    color=colors[i], linewidth=2, label=f'Curve {i+1}')
        
        ax2.set_xlabel('Age (years)')
        ax2.set_ylabel(f'{chart_type.title()} ({"cm" if chart_type == "height" else "kg"})')
        ax2.set_title(f'{chart_type.title()} Percentile Curves')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'processed_{chart_type}_curves.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main function to process manual curves."""
    print("\n" + "="*60)
    print("COORDINATE CONVERSION")
    print("="*60)
    
    # Check for manual curve files
    magnetic_file = "magnetic_curves.json"
    smart_file = "smart_selected_curves.json"
    assisted_file = "assisted_curves.json"
    tracer_file = "manual_curves.json"
    painter_file = "painted_curves.json"
    
    curves_file = None
    if Path(magnetic_file).exists():
        curves_file = magnetic_file
        print(f"Found magnetic-traced curves: {magnetic_file}")
    elif Path(smart_file).exists():
        curves_file = smart_file
        print(f"Found smart-selected curves: {smart_file}")
    elif Path(assisted_file).exists():
        curves_file = assisted_file
        print(f"Found semi-automatic curves: {assisted_file}")
    elif Path(tracer_file).exists():
        curves_file = tracer_file
        print(f"Found click-traced curves: {tracer_file}")
    elif Path(painter_file).exists():
        curves_file = painter_file
        print(f"Found painted curves: {painter_file}")
    else:
        print("No manual curve files found!")
        print("Please run the interactive tools first:")
        print("  python curve_identifier.py")
        return
    
    # Determine chart type
    print("\nWhat type of measurements are these curves for?")
    print("1. Height curves")
    print("2. Weight curves")
    chart_choice = input("Enter 1 or 2: ").strip()
    
    chart_type = 'height' if chart_choice == '1' else 'weight'
    
    # Process curves
    converter = CoordinateConverter("Weight-and-height_Boys_2-19-years.png", debug=True)
    
    try:
        chart_curves = converter.process_manual_curves(curves_file, chart_type)
        
        # Save results
        output_file = f"percentile_curves_{chart_type}.json"
        converter.save_processed_curves(chart_curves, output_file, chart_type)
        
        # Visualize results
        converter.visualize_processed_curves(chart_curves, chart_type)
        
        print(f"\nProcessing complete!")
        print(f"- Processed {len(chart_curves)} curves")
        print(f"- Saved to: {output_file}")
        print(f"- CSV format: {output_file.replace('.json', '.csv')}")
        
    except Exception as e:
        print(f"Error processing curves: {e}")


if __name__ == "__main__":
    main()