#!/usr/bin/env python3
"""
Magnetic curve tracer - intelligently follows black curves as you drag.
Simple and fast, like magnetic lasso in Photoshop.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import json
from pathlib import Path


class MagneticCurveTracer:
    """Magnetic curve tracing tool that sticks to black lines."""
    
    def __init__(self, image_path="Weight-and-height_Boys_2-19-years.png"):
        """Initialize the magnetic tracer.
        
        Args:
            image_path (str): Path to the image file
        """
        self.image_path = image_path
        
        # Generate chart name and save filename from image path
        self.chart_name = self.generate_chart_name(image_path)
        self.save_filename = self.generate_save_filename(image_path)
        
        self.original_image = None
        self.gray_image = None
        self.curves = []
        self.current_curve = []
        self.tracing = False
        self.last_point = None
        self.magnetic_strength = 15  # How far to search for black pixels
        self.fig = None
        self.ax = None
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.panning = False
        self.pan_start = None
        self.cmd_key_pressed = False
        self.pan_mode = False  # Toggle for pan mode
        self.eraser_mode = False  # Toggle for eraser mode
        self.partial_eraser_mode = False  # Toggle for partial eraser mode
        self.edit_mode = False  # Toggle for curve editing mode
        self.line_tool_mode = False  # Toggle for line tool mode
        self.selected_curve_idx = None  # Currently selected curve for editing
        self.line_start_point = None  # Start point for line tool
        self.line_end_point = None  # End point for line tool
        self.line_control_point = None  # Control point for curve adjustment
        self.dragging_control = False  # Whether we're dragging the control point
        self.last_update_time = 0  # For throttling updates
        
        # Load image
        self.load_image()
        self.setup_gui()
        
    def load_image(self):
        """Load and prepare the image."""
        # Load image
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Could not load image from {self.image_path}")
            
        # Convert BGR to RGB
        self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for processing
        self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print(f"Loaded image: {self.original_image.shape}")
        
    def setup_gui(self):
        """Set up the GUI for magnetic tracing."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 20))
        
        # Display image
        self.im_display = self.ax.imshow(self.original_image)
        self.ax.set_title(f'Magnetic Curve Tracer: {self.chart_name}')
        self.ax.axis('off')
        
        # Add buttons - Row 1
        ax_new_curve = plt.axes([0.02, 0.95, 0.06, 0.04])
        ax_finish_curve = plt.axes([0.09, 0.95, 0.06, 0.04])
        ax_delete_last = plt.axes([0.16, 0.95, 0.06, 0.04])
        ax_clear_all = plt.axes([0.23, 0.95, 0.06, 0.04])
        ax_save = plt.axes([0.30, 0.95, 0.06, 0.04])
        ax_load = plt.axes([0.37, 0.95, 0.06, 0.04])
        ax_weak_mag = plt.axes([0.44, 0.95, 0.06, 0.04])
        ax_strong_mag = plt.axes([0.51, 0.95, 0.06, 0.04])
        ax_undo = plt.axes([0.58, 0.95, 0.06, 0.04])
        ax_zoom_in = plt.axes([0.65, 0.95, 0.06, 0.04])
        ax_zoom_out = plt.axes([0.72, 0.95, 0.06, 0.04])
        ax_zoom_fit = plt.axes([0.79, 0.95, 0.06, 0.04])
        ax_pan_mode = plt.axes([0.86, 0.95, 0.06, 0.04])
        ax_eraser = plt.axes([0.93, 0.95, 0.06, 0.04])
        
        # Add buttons - Row 2 (for new editing tools)
        ax_partial_eraser = plt.axes([0.02, 0.90, 0.06, 0.04])
        ax_edit_mode = plt.axes([0.09, 0.90, 0.06, 0.04])
        ax_line_tool = plt.axes([0.16, 0.90, 0.06, 0.04])
        
        self.btn_new_curve = Button(ax_new_curve, 'New')
        self.btn_finish_curve = Button(ax_finish_curve, 'Finish')
        self.btn_delete_last = Button(ax_delete_last, 'Delete')
        self.btn_clear_all = Button(ax_clear_all, 'Clear All')
        self.btn_save = Button(ax_save, 'Save')
        self.btn_load = Button(ax_load, 'Load')
        self.btn_weak_mag = Button(ax_weak_mag, 'Weak Mag')
        self.btn_strong_mag = Button(ax_strong_mag, 'Strong Mag')
        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_zoom_in = Button(ax_zoom_in, 'Zoom +')
        self.btn_zoom_out = Button(ax_zoom_out, 'Zoom -')
        self.btn_zoom_fit = Button(ax_zoom_fit, 'Fit')
        self.btn_pan_mode = Button(ax_pan_mode, 'Pan')
        self.btn_eraser = Button(ax_eraser, 'Eraser')
        
        # New editing tool buttons
        self.btn_partial_eraser = Button(ax_partial_eraser, 'PartErase')
        self.btn_edit_mode = Button(ax_edit_mode, 'Edit')
        self.btn_line_tool = Button(ax_line_tool, 'Line Tool')
        
        # Connect button events
        self.btn_new_curve.on_clicked(self.start_new_curve)
        self.btn_finish_curve.on_clicked(self.finish_current_curve)
        self.btn_delete_last.on_clicked(self.delete_last_points)
        self.btn_clear_all.on_clicked(self.clear_all_curves)
        self.btn_save.on_clicked(self.save_curves)
        self.btn_load.on_clicked(self.load_curves)
        self.btn_weak_mag.on_clicked(lambda x: self.set_magnetic_strength(8))
        self.btn_strong_mag.on_clicked(lambda x: self.set_magnetic_strength(25))
        self.btn_undo.on_clicked(self.undo_last_curve)
        self.btn_zoom_in.on_clicked(self.zoom_in)
        self.btn_zoom_out.on_clicked(self.zoom_out)
        self.btn_zoom_fit.on_clicked(self.zoom_fit)
        self.btn_pan_mode.on_clicked(self.toggle_pan_mode)
        self.btn_eraser.on_clicked(self.toggle_eraser_mode)
        
        # Connect new editing tool events
        self.btn_partial_eraser.on_clicked(self.toggle_partial_eraser_mode)
        self.btn_edit_mode.on_clicked(self.toggle_edit_mode)
        self.btn_line_tool.on_clicked(self.toggle_line_tool_mode)
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        
        # Status text
        self.status_text = self.fig.text(0.02, 0.02, 
                                        f'Curves: {len(self.curves)} | Current: {len(self.current_curve)} | Magnetic: {self.magnetic_strength}px | Zoom: {self.zoom_factor:.1f}x | Target: 14', 
                                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        self.update_display()
        
    def generate_chart_name(self, image_path):
        """Generate a readable chart name from the image path."""
        # Extract filename without extension
        filename = Path(image_path).stem
        
        # Clean up the filename for display
        chart_name = filename.replace('-', ' ').replace('_', ' ')
        
        return chart_name
        
    def generate_save_filename(self, image_path):
        """Generate the JSON save filename from the image path."""
        # Extract filename without extension
        filename = Path(image_path).stem
        
        # Convert to lowercase and replace spaces/dashes with underscores
        save_name = filename.lower().replace(' ', '_').replace('-', '_')
        
        return f"{save_name}_curves.json"
        
    def set_magnetic_strength(self, strength):
        """Set the magnetic attraction strength."""
        self.magnetic_strength = strength
        print(f"🧲 Magnetic strength set to {strength} pixels")
        self.update_status()
        
    def on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:  # Left click
            x, y = int(event.xdata), int(event.ydata)
            
            if self.eraser_mode:
                # Eraser mode - delete entire curve at clicked location
                self.delete_curve_at_point(x, y)
            elif self.partial_eraser_mode:
                # Partial eraser mode - erase part of curve
                self.partial_erase_curve(x, y)
            elif self.edit_mode:
                # Edit mode - select curve for editing
                self.select_curve_for_editing(x, y)
            elif self.line_tool_mode:
                # Line tool mode - handle point placement
                if self.line_start_point is None:
                    # Set start point
                    self.line_start_point = [x, y]
                    print("📐 Line tool: Point A set, click Point B")
                elif self.line_end_point is None:
                    # Set end point and calculate middle control point
                    self.line_end_point = [x, y]
                    # Initialize control point at midpoint
                    mid_x = (self.line_start_point[0] + self.line_end_point[0]) // 2
                    mid_y = (self.line_start_point[1] + self.line_end_point[1]) // 2
                    self.line_control_point = [mid_x, mid_y]
                    print("📐 Line tool: Point B set, drag the control point to adjust curve")
                elif self.line_control_point:
                    # Check if clicking near control point to drag it
                    control_dist = np.sqrt((x - self.line_control_point[0])**2 + (y - self.line_control_point[1])**2)
                    if control_dist <= 20:  # Close to control point
                        self.dragging_control = True
                        print("📐 Dragging control point")
                    else:
                        # Finish the line tool curve
                        self.add_line_tool_curve()
                self.update_display()
            elif self.pan_mode or self.cmd_key_pressed:
                # Pan mode enabled via button or Command key
                self.panning = True
                self.pan_start = (event.xdata, event.ydata)
                print("Pan mode: drag to pan")
            else:  # Normal left click = trace curve
                self.tracing = True
                
                # Find magnetic point near click
                magnetic_point = self.find_magnetic_point(x, y)
                
                self.current_curve.append(magnetic_point)
                self.last_point = magnetic_point
                
                print(f"Started tracing at: {magnetic_point}")
                self.update_display()
            
        elif event.button == 3:  # Right click - start panning
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)
            
    def on_release(self, event):
        """Handle mouse release events."""
        if event.button == 1:  # Left click release
            if self.dragging_control:
                # Stop dragging control point
                self.dragging_control = False
                print("📐 Control point adjusted")
            elif self.panning:  # Was panning with Command + left click
                self.panning = False
                self.pan_start = None
            else:  # Was tracing
                self.tracing = False
                print(f"Stopped tracing. Current curve has {len(self.current_curve)} points")
        elif event.button == 3:  # Right click release
            self.panning = False
            self.pan_start = None
        
    def on_motion(self, event):
        """Handle mouse motion events for magnetic tracing and panning."""
        if event.inaxes != self.ax:
            return
            
        # Throttle updates to reduce stuttering
        import time
        current_time = time.time()
        if current_time - self.last_update_time < 0.016:  # ~60 FPS limit
            return
        self.last_update_time = current_time
            
        if self.dragging_control and self.line_tool_mode:
            # Update control point position
            x, y = int(event.xdata), int(event.ydata)
            self.line_control_point = [x, y]
            self.update_display()
        elif self.tracing:
            x, y = int(event.xdata), int(event.ydata)
            
            # Only add point if mouse moved significantly
            if self.last_point is not None:
                distance = np.sqrt((x - self.last_point[0])**2 + (y - self.last_point[1])**2)
                if distance < 5:  # Too close to last point
                    return
            
            # Find magnetic point near cursor
            magnetic_point = self.find_magnetic_point(x, y)
            
            # Add intermediate magnetic points if the jump is too large
            if self.last_point is not None:
                distance = np.sqrt((magnetic_point[0] - self.last_point[0])**2 + 
                                 (magnetic_point[1] - self.last_point[1])**2)
                
                if distance > 20:  # Large jump, add intermediate points
                    intermediate_points = self.get_intermediate_magnetic_points(self.last_point, magnetic_point)
                    self.current_curve.extend(intermediate_points)
                else:
                    self.current_curve.append(magnetic_point)
            else:
                self.current_curve.append(magnetic_point)
                
            self.last_point = magnetic_point
            self.update_display()
            
        elif self.panning and self.pan_start:
            # Handle panning - calculate movement in image coordinates
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            
            # Apply movement with bounds checking
            self.pan_x -= dx
            self.pan_y -= dy
            
            # Clamp pan to image bounds
            self.clamp_pan_to_bounds()
            
            # Don't update pan_start to avoid accumulation
            self.update_display()
            
    def on_key_press(self, event):
        """Handle key press events."""
        # Handle Command key (cmd on Mac, ctrl on PC)
        if event.key in ['cmd+super', 'super', 'ctrl', 'control']:
            self.cmd_key_pressed = True
            print("Command key pressed - pan mode enabled")
            
    def on_key_release(self, event):
        """Handle key release events."""
        # Handle Command key (cmd on Mac, ctrl on PC)
        if event.key in ['cmd+super', 'super', 'ctrl', 'control']:
            self.cmd_key_pressed = False
            print("Command key released - trace mode enabled")
        
    def find_magnetic_point(self, x, y):
        """Find the nearest dark point (curve) to the given coordinates."""
        # Search in a small area around the click point
        search_radius = self.magnetic_strength
        
        # Get image dimensions
        h, w = self.gray_image.shape
        
        # Define search area
        x_min = max(0, x - search_radius)
        x_max = min(w, x + search_radius)
        y_min = max(0, y - search_radius)
        y_max = min(h, y + search_radius)
        
        # Extract search region
        search_region = self.gray_image[y_min:y_max, x_min:x_max]
        
        if search_region.size == 0:
            return [x, y]
        
        # Find darkest points in the region (curves are dark)
        dark_threshold = 80  # Adjust based on your image
        dark_mask = search_region < dark_threshold
        
        if not np.any(dark_mask):
            # No dark pixels found, return original point
            return [x, y]
        
        # Find coordinates of dark pixels
        dark_y, dark_x = np.where(dark_mask)
        
        # Convert back to full image coordinates
        dark_x_full = dark_x + x_min
        dark_y_full = dark_y + y_min
        
        # Find the closest dark pixel to the click point
        distances = (dark_x_full - x)**2 + (dark_y_full - y)**2
        closest_idx = np.argmin(distances)
        
        magnetic_x = dark_x_full[closest_idx]
        magnetic_y = dark_y_full[closest_idx]
        
        return [magnetic_x, magnetic_y]
        
    def get_intermediate_magnetic_points(self, start_point, end_point):
        """Get intermediate magnetic points between two points."""
        start_x, start_y = start_point
        end_x, end_y = end_point
        
        # Calculate number of intermediate points needed
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        num_points = max(1, int(distance / 10))  # One point every 10 pixels
        
        intermediate_points = []
        
        for i in range(1, num_points):
            # Linear interpolation
            t = i / num_points
            interp_x = int(start_x + t * (end_x - start_x))
            interp_y = int(start_y + t * (end_y - start_y))
            
            # Find magnetic point near interpolated position
            magnetic_point = self.find_magnetic_point(interp_x, interp_y)
            intermediate_points.append(magnetic_point)
        
        # Add the final point
        intermediate_points.append(end_point)
        
        return intermediate_points
        
    def start_new_curve(self, event):
        """Start tracing a new curve."""
        if self.current_curve:
            self.finish_current_curve(None)
        
        self.current_curve = []
        self.last_point = None
        print(f"Started new curve #{len(self.curves) + 1}")
        self.update_display()
        
    def finish_current_curve(self, event):
        """Finish the current curve and add it to the list."""
        if len(self.current_curve) >= 10:  # Minimum points for a valid curve
            # Smooth the curve slightly
            smoothed_curve = self.smooth_curve(self.current_curve)
            self.curves.append(smoothed_curve)
            self.current_curve = []
            self.last_point = None
            print(f"Finished curve #{len(self.curves)} with {len(smoothed_curve)} points")
        else:
            print(f"Curve too short ({len(self.current_curve)} points). Need at least 10 points.")
            
        self.update_display()
        
    def smooth_curve(self, curve_points):
        """Apply light smoothing to the curve."""
        if len(curve_points) < 5:
            return curve_points
            
        # Convert to numpy array
        points = np.array(curve_points)
        
        # Simple moving average smoothing
        window_size = 3
        smoothed_points = []
        
        for i in range(len(points)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(points), i + window_size // 2 + 1)
            
            avg_x = np.mean(points[start_idx:end_idx, 0])
            avg_y = np.mean(points[start_idx:end_idx, 1])
            
            smoothed_points.append([int(avg_x), int(avg_y)])
        
        return smoothed_points
        
    def delete_last_points(self, event):
        """Delete the last few points from current curve."""
        if self.current_curve:
            # Remove last 5 points or all if fewer than 5
            points_to_remove = min(5, len(self.current_curve))
            for _ in range(points_to_remove):
                if self.current_curve:
                    removed = self.current_curve.pop()
                    
            if self.current_curve:
                self.last_point = self.current_curve[-1]
            else:
                self.last_point = None
                
            print(f"Removed {points_to_remove} points")
            self.update_display()
            
    def undo_last_curve(self, event):
        """Undo the last completed curve."""
        if self.curves:
            removed_curve = self.curves.pop()
            print(f"Undid curve with {len(removed_curve)} points")
            self.update_display()
            
    def clear_all_curves(self, event):
        """Clear all curves and start over."""
        self.curves = []
        self.current_curve = []
        self.last_point = None
        print("Cleared all curves")
        self.update_display()
        
    def update_display(self):
        """Update the display with current curves, zoom, and pan."""
        # Reset to original image
        display_image = self.original_image.copy()
        
        # Draw completed curves
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(self.curves), 1)))
        
        for i, curve in enumerate(self.curves):
            color = (int(colors[i][0]*255), int(colors[i][1]*255), int(colors[i][2]*255))
            
            # Highlight selected curve in edit mode
            line_thickness = 5 if (self.edit_mode and i == self.selected_curve_idx) else 3
            
            # Draw curve line
            for j in range(len(curve) - 1):
                cv2.line(display_image, 
                        tuple(curve[j]), 
                        tuple(curve[j+1]), 
                        color, line_thickness)
            
            # Draw curve number
            if len(curve) > 0:
                mid_idx = len(curve) // 2
                cv2.putText(display_image, str(i+1), 
                           tuple(curve[mid_idx]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (255, 255, 255), 3)
                cv2.putText(display_image, str(i+1), 
                           tuple(curve[mid_idx]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           color, 2)
        
        # Draw current curve being traced (in red)
        if len(self.current_curve) > 1:
            for j in range(len(self.current_curve) - 1):
                cv2.line(display_image, 
                        tuple(self.current_curve[j]), 
                        tuple(self.current_curve[j+1]), 
                        (255, 0, 0), 2)
        
        # Draw current curve points
        for point in self.current_curve:
            cv2.circle(display_image, tuple(point), 2, (255, 0, 0), -1)
        
        # Draw line tool visualization
        if self.line_tool_mode:
            if self.line_start_point:
                # Draw start point
                cv2.circle(display_image, tuple(self.line_start_point), 8, (0, 255, 0), -1)
                cv2.putText(display_image, 'A', 
                           (self.line_start_point[0] + 10, self.line_start_point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.line_end_point:
                # Draw end point
                cv2.circle(display_image, tuple(self.line_end_point), 8, (0, 255, 0), -1)
                cv2.putText(display_image, 'B', 
                           (self.line_end_point[0] + 10, self.line_end_point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.line_control_point and self.line_start_point and self.line_end_point:
                # Draw control point
                cv2.circle(display_image, tuple(self.line_control_point), 10, (255, 0, 255), -1)
                cv2.circle(display_image, tuple(self.line_control_point), 12, (255, 255, 255), 2)
                
                # Draw control lines
                cv2.line(display_image, tuple(self.line_start_point), tuple(self.line_control_point), (200, 200, 200), 1)
                cv2.line(display_image, tuple(self.line_control_point), tuple(self.line_end_point), (200, 200, 200), 1)
                
                # Draw preview curve
                preview_curve = self.generate_curve_from_line_tool()
                if len(preview_curve) > 1:
                    for j in range(len(preview_curve) - 1):
                        cv2.line(display_image, 
                                tuple(preview_curve[j]), 
                                tuple(preview_curve[j+1]), 
                                (255, 0, 255), 2)
        
        # Update display with zoom and pan
        self.im_display.set_array(display_image)
        
        # Calculate view limits based on zoom and pan
        h, w = display_image.shape[:2]
        
        # Calculate zoomed view size
        view_w = w / self.zoom_factor
        view_h = h / self.zoom_factor
        
        # Calculate center point with pan offset
        center_x = w/2 + self.pan_x
        center_y = h/2 + self.pan_y
        
        # Calculate view bounds
        x_min = center_x - view_w/2
        x_max = center_x + view_w/2
        y_min = center_y - view_h/2
        y_max = center_y + view_h/2
        
        # Apply zoom/pan to axes (note: y is inverted for image coordinates)
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_max, y_min)  # Inverted for image coordinates
        
        self.update_status()
        
        # Use blit for faster updates during panning/zooming
        if hasattr(self, '_background'):
            self.fig.canvas.restore_region(self._background)
        
        self.fig.canvas.draw_idle()  # Use draw_idle instead of draw for better performance
        
    def update_status(self):
        """Update the status text."""
        status_color = "lightgreen" if len(self.curves) == 14 else "lightyellow"
        zoom_percent = int(self.zoom_factor * 100)
        
        # Add mode indicator
        mode_text = ""
        if self.line_tool_mode:
            mode_text = " | Mode: LINE TOOL 📐"
            status_color = "plum"
        elif self.edit_mode:
            mode_text = " | Mode: EDIT ✏️"
            if self.selected_curve_idx is not None:
                mode_text += f" (Curve #{self.selected_curve_idx + 1})"
            status_color = "lightblue"
        elif self.partial_eraser_mode:
            mode_text = " | Mode: PARTIAL ERASE ✂️"
            status_color = "orange"
        elif self.eraser_mode:
            mode_text = " | Mode: ERASER 🗑️"
            status_color = "lightcoral"
        elif self.pan_mode:
            mode_text = " | Mode: PAN ✋"
            status_color = "lightblue"
        else:
            mode_text = " | Mode: TRACE ✏️"
        
        self.status_text.set_text(
            f'Curves: {len(self.curves)} | Current: {len(self.current_curve)} | Magnetic: {self.magnetic_strength}px | Zoom: {zoom_percent}%{mode_text} | Target: 14'
        )
        self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=status_color))
        
    def save_curves(self, event):
        """Save curves to JSON file."""
        if not self.curves:
            print("No curves to save")
            return
            
        # Finish current curve if it has points
        if len(self.current_curve) >= 10:
            self.finish_current_curve(None)
        
        data = {
            'chart_name': self.chart_name,
            'image_path': self.image_path,
            'image_shape': self.original_image.shape,
            'curves': self.curves,
            'num_curves': len(self.curves),
            'method': 'magnetic_tracing'
        }
        
        filename = self.save_filename
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"💾 Saved {len(self.curves)} curves to {filename}")
        
    def load_curves(self, event):
        """Load curves from JSON file."""
        filename = self.save_filename
        if not Path(filename).exists():
            print(f"File {filename} not found")
            return
            
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            self.curves = data['curves']
            self.current_curve = []
            self.last_point = None
            print(f"📂 Loaded {len(self.curves)} curves from {filename}")
            self.update_display()
            
        except Exception as e:
            print(f"Error loading curves: {e}")
    
    def run(self):
        """Start the magnetic tracing session."""
        print("\n" + "🧲" + "="*70)
        print("MAGNETIC CURVE TRACER")
        print("="*70 + "🧲")
        print(f"📊 Chart: {self.chart_name}")
        print(f"💾 Saves to: {self.save_filename}")
        print("="*70)
        print("Instructions:")
        print("1. Click 'New' to start a new curve")
        print("2. CLICK AND DRAG along a curve - it will snap to black lines!")
        print("3. Release mouse when done with that segment")
        print("4. Continue clicking and dragging to extend the curve")
        print("5. Click 'Finish' when the curve is complete")
        print("6. Repeat for all percentile curves")
        print()
        print("🧲 Magnetic Controls:")
        print("• 'Weak Mag' = Less magnetic attraction (more freedom)")
        print("• 'Strong Mag' = Strong magnetic attraction (sticks tight)")
        print("• 'Delete' = Remove last few points if you make a mistake")
        print("• 'Undo' = Remove the last completed curve")
        print()
        print("�️ Eraser Tool:")
        print("• 'Eraser' button = Toggle eraser mode")
        print("• In eraser mode: Click on any curve to delete it")
        print("• Status bar shows current mode (TRACE/PAN/ERASER)")
        print()
        print("�🔍 Zoom Controls:")
        print("• 'Zoom +' = Zoom in for more precision")
        print("• 'Zoom -' = Zoom out for overview")
        print("• 'Fit' = Reset zoom to show entire image")
        print("• Mouse wheel = Quick zoom in/out")
        print("• 'Pan' button = Toggle pan mode (click to pan instead of trace)")
        print("• Command + click and drag = Pan around zoomed image")
        print("• Right-click and drag = Pan around zoomed image")
        print()
        print("🎯 Goal: Trace all percentile curves")
        print("The cursor will automatically snap to nearby black lines!")
        print(f"💾 Your progress saves to: {self.save_filename}")
        print("="*70)
        
        plt.show()
        
        return self.curves

    def zoom_in(self, event):
        """Zoom in to the image."""
        if self.zoom_factor < 10.0:  # Limit maximum zoom
            self.zoom_factor *= 1.25  # Smaller increment for smoother zooming
            self.clamp_pan_to_bounds()
            self.update_display()
        
    def zoom_out(self, event):
        """Zoom out from the image."""
        if self.zoom_factor > 0.1:  # Limit minimum zoom
            self.zoom_factor /= 1.25  # Smaller increment for smoother zooming
            self.clamp_pan_to_bounds()
            self.update_display()
        
    def zoom_fit(self, event):
        """Reset zoom to fit entire image."""
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_display()
        
    def on_scroll(self, event):
        """Handle scroll wheel zoom."""
        if event.inaxes != self.ax:
            return
            
        # Get the current mouse position in axes coordinates
        mouse_x, mouse_y = event.xdata, event.ydata
        if mouse_x is None or mouse_y is None:
            return
        
        # Calculate zoom direction and amount
        if event.step > 0:  # Scroll up - zoom in
            if self.zoom_factor >= 10.0:
                return  # Max zoom reached
            zoom_scale = 1.15
        else:  # Scroll down - zoom out
            if self.zoom_factor <= 0.1:
                return  # Min zoom reached
            zoom_scale = 1.0 / 1.15
        
        # Store old zoom for pan adjustment
        old_zoom = self.zoom_factor
        self.zoom_factor *= zoom_scale
        
        # Simple pan adjustment to keep zoom centered on mouse
        h, w = self.original_image.shape[:2]
        
        # Calculate current view center in image coordinates
        current_center_x = w/2 + self.pan_x
        current_center_y = h/2 + self.pan_y
        
        # Calculate mouse position relative to image center
        mouse_offset_x = mouse_x - w/2
        mouse_offset_y = mouse_y - h/2
        
        # Adjust pan to keep mouse position relatively fixed
        zoom_change = (self.zoom_factor - old_zoom) / old_zoom
        self.pan_x += mouse_offset_x * zoom_change * 0.5
        self.pan_y += mouse_offset_y * zoom_change * 0.5
        
        # Ensure we stay within bounds
        self.clamp_pan_to_bounds()
        self.update_display()

    def clamp_pan_to_bounds(self):
        """Ensure pan coordinates stay within image bounds."""
        h, w = self.original_image.shape[:2]
        
        # Calculate the visible area dimensions at current zoom
        view_w = w / self.zoom_factor
        view_h = h / self.zoom_factor
        
        # Calculate maximum pan offsets (half the difference between image and view)
        max_pan_x = max(0, (w - view_w) / 2)
        max_pan_y = max(0, (h - view_h) / 2)
        
        # Clamp pan values
        self.pan_x = max(-max_pan_x, min(max_pan_x, self.pan_x))
        self.pan_y = max(-max_pan_y, min(max_pan_y, self.pan_y))

    def toggle_pan_mode(self, event):
        """Toggle pan mode on/off."""
        self.pan_mode = not self.pan_mode
        mode_text = "ON" if self.pan_mode else "OFF"
        print(f"Pan mode: {mode_text}")
        
        if self.pan_mode:
            # Clear other modes
            self.clear_all_modes()
            self.pan_mode = True
            self.btn_pan_mode.label.set_text('Pan✓')
            self.btn_pan_mode.color = 'lightgreen'
        else:
            self.btn_pan_mode.label.set_text('Pan')
            self.btn_pan_mode.color = 'lightgray'
        
        self.update_status()
        self.fig.canvas.draw()

    def toggle_eraser_mode(self, event):
        """Toggle eraser mode on/off."""
        self.eraser_mode = not self.eraser_mode
        mode_text = "ON" if self.eraser_mode else "OFF"
        print(f"🗑️ Eraser mode: {mode_text}")
        
        if self.eraser_mode:
            # Clear other modes
            self.clear_all_modes()
            self.eraser_mode = True
            self.btn_eraser.label.set_text('Erase✓')
            self.btn_eraser.color = 'lightcoral'
        else:
            self.btn_eraser.label.set_text('Eraser')
            self.btn_eraser.color = 'lightgray'
        
        self.update_status()
        self.fig.canvas.draw()

    def clear_all_modes(self):
        """Clear all editing modes."""
        self.pan_mode = False
        self.eraser_mode = False
        self.partial_eraser_mode = False
        self.edit_mode = False
        self.line_tool_mode = False
        self.selected_curve_idx = None
        self.line_start_point = None
        self.line_end_point = None
        self.line_control_point = None
        
        # Reset all button appearances
        self.btn_pan_mode.label.set_text('Pan')
        self.btn_pan_mode.color = 'lightgray'
        self.btn_eraser.label.set_text('Eraser')
        self.btn_eraser.color = 'lightgray'
        self.btn_partial_eraser.label.set_text('PartErase')
        self.btn_partial_eraser.color = 'lightgray'
        self.btn_edit_mode.label.set_text('Edit')
        self.btn_edit_mode.color = 'lightgray'
        self.btn_line_tool.label.set_text('Line Tool')
        self.btn_line_tool.color = 'lightgray'

    def toggle_partial_eraser_mode(self, event):
        """Toggle partial eraser mode on/off."""
        self.partial_eraser_mode = not self.partial_eraser_mode
        mode_text = "ON" if self.partial_eraser_mode else "OFF"
        print(f"✂️ Partial eraser mode: {mode_text}")
        
        if self.partial_eraser_mode:
            # Clear other modes
            self.clear_all_modes()
            self.partial_eraser_mode = True
            self.btn_partial_eraser.label.set_text('PartErase✓')
            self.btn_partial_eraser.color = 'orange'
        else:
            self.btn_partial_eraser.label.set_text('PartErase')
            self.btn_partial_eraser.color = 'lightgray'
        
        self.update_status()
        self.fig.canvas.draw()

    def toggle_edit_mode(self, event):
        """Toggle curve editing mode on/off."""
        self.edit_mode = not self.edit_mode
        mode_text = "ON" if self.edit_mode else "OFF"
        print(f"✏️ Edit mode: {mode_text}")
        
        if self.edit_mode:
            # Clear other modes
            self.clear_all_modes()
            self.edit_mode = True
            self.btn_edit_mode.label.set_text('Edit✓')
            self.btn_edit_mode.color = 'lightblue'
            print("Click on a curve to select it for editing")
        else:
            self.btn_edit_mode.label.set_text('Edit')
            self.btn_edit_mode.color = 'lightgray'
            self.selected_curve_idx = None
        
        self.update_status()
        self.fig.canvas.draw()

    def toggle_line_tool_mode(self, event):
        """Toggle line tool mode on/off."""
        self.line_tool_mode = not self.line_tool_mode
        mode_text = "ON" if self.line_tool_mode else "OFF"
        print(f"📐 Line tool mode: {mode_text}")
        
        if self.line_tool_mode:
            # Clear other modes
            self.clear_all_modes()
            self.line_tool_mode = True
            self.btn_line_tool.label.set_text('Line Tool✓')
            self.btn_line_tool.color = 'purple'
            print("Click point A, then point B, then adjust the middle control point")
        else:
            self.btn_line_tool.label.set_text('Line Tool')
            self.btn_line_tool.color = 'lightgray'
            self.line_start_point = None
            self.line_end_point = None
            self.line_control_point = None
        
        self.update_status()
        self.fig.canvas.draw()

    def find_curve_near_point(self, x, y, tolerance=20):
        """Find which curve (if any) is near the clicked point."""
        click_point = np.array([x, y])
        
        for curve_idx, curve in enumerate(self.curves):
            curve_array = np.array(curve)
            
            # Calculate distances from click point to all points in the curve
            distances = np.sqrt(np.sum((curve_array - click_point)**2, axis=1))
            min_distance = np.min(distances)
            
            # If any point in the curve is within tolerance, return this curve
            if min_distance <= tolerance:
                return curve_idx
        
        return None

    def delete_curve_at_point(self, x, y):
        """Delete the curve that was clicked on."""
        curve_idx = self.find_curve_near_point(x, y)
        
        if curve_idx is not None:
            deleted_curve = self.curves.pop(curve_idx)
            print(f"🗑️ Deleted curve #{curve_idx + 1} with {len(deleted_curve)} points")
            self.update_display()
            return True
        else:
            print("🗑️ No curve found at clicked location")
            return False

    def partial_erase_curve(self, x, y, erase_radius=30):
        """Erase part of a curve around the clicked point like a normal eraser."""
        click_point = np.array([x, y])
        curves_modified = False
        
        for curve_idx, curve in enumerate(self.curves):
            curve_array = np.array(curve)
            
            # Find points within erase radius
            distances = np.sqrt(np.sum((curve_array - click_point)**2, axis=1))
            points_to_remove = distances <= erase_radius
            
            # If some points should be removed
            if np.any(points_to_remove):
                # Find continuous segments to remove
                remove_indices = np.where(points_to_remove)[0]
                
                if len(remove_indices) > 0:
                    # Split curve at removal points
                    new_curves = []
                    current_segment = []
                    
                    for i, point in enumerate(curve):
                        if i not in remove_indices:
                            current_segment.append(point)
                        else:
                            # End current segment if it has enough points
                            if len(current_segment) >= 10:
                                new_curves.append(current_segment)
                            current_segment = []
                    
                    # Add final segment if it has enough points
                    if len(current_segment) >= 10:
                        new_curves.append(current_segment)
                    
                    # Update curves list
                    self.curves.pop(curve_idx)
                    
                    # Add new segments as separate curves
                    for new_curve in new_curves:
                        self.curves.insert(curve_idx, new_curve)
                        curve_idx += 1
                    
                    erased_count = len(remove_indices)
                    segments_created = len(new_curves)
                    print(f"✂️ Erased {erased_count} points from curve, created {segments_created} segments")
                    
                    curves_modified = True
                    break  # Only modify one curve per click
        
        if curves_modified:
            self.update_display()
        else:
            print("✂️ No curve points found within erase radius")
        
        return curves_modified

    def select_curve_for_editing(self, x, y):
        """Select a curve for editing and enable tracing mode for that curve."""
        curve_idx = self.find_curve_near_point(x, y)
        
        if curve_idx is not None:
            self.selected_curve_idx = curve_idx
            # Move selected curve to current_curve for editing
            self.current_curve = self.curves[curve_idx].copy()
            self.last_point = self.current_curve[-1] if self.current_curve else None
            print(f"✏️ Selected curve #{curve_idx + 1} for editing - you can now trace to extend it")
            print("💡 Use normal tracing (click and drag) to add to this curve")
            self.update_display()
            return True
        else:
            print("✏️ No curve found at clicked location")
            return False

    def generate_curve_from_line_tool(self):
        """Generate a smooth curve from line tool points."""
        if not (self.line_start_point and self.line_end_point and self.line_control_point):
            return []
        
        # Use quadratic Bezier curve
        start = np.array(self.line_start_point)
        control = np.array(self.line_control_point)
        end = np.array(self.line_end_point)
        
        # Generate curve points
        t_values = np.linspace(0, 1, 50)  # 50 points for smooth curve
        curve_points = []
        
        for t in t_values:
            # Quadratic Bezier formula: (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
            point = (1-t)**2 * start + 2*(1-t)*t * control + t**2 * end
            curve_points.append([int(point[0]), int(point[1])])
        
        return curve_points

    def add_line_tool_curve(self):
        """Add the line tool curve to the current curve being traced."""
        new_curve_segment = self.generate_curve_from_line_tool()
        if new_curve_segment:
            # Add to current curve instead of creating a new one
            if len(self.current_curve) == 0:
                # If no current curve, start a new one
                self.current_curve = new_curve_segment
                print(f"📐 Started new curve with line tool ({len(new_curve_segment)} points)")
            else:
                # Extend the current curve
                self.current_curve.extend(new_curve_segment)
                print(f"📐 Added line segment to current curve ({len(new_curve_segment)} points)")
            
            # Update last point for continued tracing
            self.last_point = self.current_curve[-1] if self.current_curve else None
            
            # Reset line tool
            self.line_start_point = None
            self.line_end_point = None
            self.line_control_point = None
            
            self.update_display()
            return True
        return False


if __name__ == "__main__":
    import sys
    
    # Get image path from command line argument or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "Weight-and-height_Boys_2-19-years.png"
        
    print(f"🚀 Starting Magnetic Curve Tracer for: {image_path}")
    
    tracer = MagneticCurveTracer(image_path)
    curves = tracer.run()
    
    print(f"\n🎉 Final result: {len(curves)} curves traced")
    print(f"💾 Saved to: {tracer.save_filename}")
    for i, curve in enumerate(curves):
        print(f"   Curve {i+1}: {len(curve)} points")