import random
# import time
# from tkinter.constants import COMMAND
from PIL import ImageTk
import tkinter as tk


class UI:

    def __init__(self, anim, cam):
        # MAIN WINDOW
        self.main_window = tk.Tk()
        self.main_window.title("Face Tests")
        self.main_window.geometry("1600x1050")

        # ANIMATION 
        self.animation = anim
        self.animation_panel = tk.Label(self.main_window)
        self.animation.set_tk_panel(self.animation_panel)

        # WEBCAM 
        self.camera = cam
        self.webcam_panel = tk.Label(self.main_window)
        self.camera.set_tk_panel(self.webcam_panel)

        # CLOSING WINDOW ROUTINE
        self.main_window.protocol("WM_DELETE_WINDOW", self.__on_closing)

        # DIRECTIONS
        # Get directions list
        self.directions = self.animation.get_directions()        

        # Scrollable frame
        self.container = tk.Frame(self.main_window)
        self.canvas = tk.Canvas(self.container)
        self.scrollbar = tk.Scrollbar(self.container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Directions controls
        self.directions_reset_all =  tk.Button(self.scrollable_frame, text="reset all", command=self.reset_all_directions)
        self.directions_reset_all.grid(row=0, column=0)
        self.save_directions = tk.Entry(self.scrollable_frame)
        self.save_directions.grid(row=0, column=1)
        self.save_directions_button = tk.Button(self.scrollable_frame, text="Export", command=self.export_directions)
        self.save_directions_button.grid(row=0, column=2)

        self.directions_reset =      [tk.Button(self.scrollable_frame, text="reset") for i in range(len(self.directions))]
        self.sliders_directions =    [tk.Scale( self.scrollable_frame, 
                                                orient='horizontal', 
                                                length=200, 
                                                from_=-20.0, 
                                                to_=20.0, 
                                                resolution=0.01) for i in range(len(self.directions))]
        self.directions_values =     [tk.DoubleVar(self.scrollable_frame) for i in range(len(self.directions))]
        self.directions_labels =     [tk.Label(self.scrollable_frame) for i in range(len(self.directions))]

        for direction in self.directions:
            direction_index = list(self.directions.keys()).index(direction)

            self.directions_reset[direction_index].configure(command=lambda button_id=direction_index: self.directions_reset_pressed(button_id))
            self.directions_reset[direction_index].grid(row=direction_index + 1, column=0)

            self.sliders_directions[direction_index].configure(variable=self.directions_values[direction_index], command=lambda value, dir=direction: self.sliders_directions_changed(value, dir))
            self.sliders_directions[direction_index].grid(row=direction_index + 1, column=1)

            self.directions_labels[direction_index].configure(text=direction)
            self.directions_labels[direction_index].grid(row=direction_index + 1, column=2, sticky="W")

        self.container.grid(row=0, 
                            column=0, 
                            sticky="nsew")
        self.canvas.pack(side="left", fill="both", expand=True, padx=(0, 15))
        self.scrollbar.pack(side="right", fill="y")

        # ENCODED IMAGE
        self.encoded_panel = tk.Label(self.main_window)
        self.encoded_panel.grid(row=0, 
                                column=1, 
                                sticky="NW")

        # ANIMATION
        self.animation_panel.grid(  row=self.encoded_panel.grid_info()['row'], 
                                    column=self.encoded_panel.grid_info()['column']+1, 
                                    sticky="NW")
        self.animation_button = tk.Button(self.main_window, text="INANIMATION", relief="raised", command=self.toggle_animation)
        self.animation_button.grid( row= self.animation_panel.grid_info()['row'], 
                                    column=self.animation_panel.grid_info()['column'], 
                                    sticky="NW")

        # ANIMATION PARAMETERS
        self.anim_container = tk.Frame(self.main_window)
        self.anim_container.grid(   row=self.animation_panel.grid_info()['row'], 
                                    column=self.animation_panel.grid_info()['column']+1, 
                                    sticky="NW")
        # Change face
        self.random_face_button = tk.Button(self.anim_container, text="Random Face", command=self.random_face)
        self.random_face_button.grid(   row=0, 
                                        column=0, 
                                        sticky="NW")                           
        self.encoded_face_button = tk.Button(self.anim_container, text="Encoded Face", command=self.encoded_face)
        self.encoded_face_button.grid(  row=0, 
                                        column=1, 
                                        sticky="NW")
        # Jitter: Amplitude
        self.jitter_amp_val = tk.DoubleVar(self.main_window)
        self.jitter_amp = tk.Scale(  self.anim_container, 
                                orient='horizontal', 
                                length=200, 
                                from_=0.0, 
                                to_=1.0, 
                                resolution=0.01,
                                variable=self.jitter_amp_val,
                                command=self.jitter_amp_changed)
        self.jitter_amp.grid(   row=1, 
                                column=0, 
                                sticky="NW")
        self.jitter_amp_label = tk.Label(self.anim_container, text="Jitter amp")
        self.jitter_amp_label.grid( row=1, 
                                    column=1, 
                                    sticky="NW")
        self.jitter_amp.set(0.3)
        # Jitter: Speed
        self.jitter_speed_val = tk.DoubleVar(self.main_window)
        self.jitter_speed = tk.Scale(   self.anim_container, 
                                        orient='horizontal', 
                                        length=200, 
                                        from_=0.0, 
                                        to_=1.0, 
                                        resolution=0.01,
                                        variable=self.jitter_speed_val,
                                        command=self.jitter_speed_changed)
        self.jitter_speed.grid( row=2, 
                                column=0, 
                                sticky="NW")
        self.jitter_speed_label = tk.Label(self.anim_container, text="Jitter speed")
        self.jitter_speed_label.grid(   row=2, 
                                        column=1, 
                                        sticky="NW")
        self.jitter_speed.set(0.03)
        # Save animation latents
        self.save_anim_latents_button = tk.Button(self.anim_container, text="Save unreal", command=self.save_anim_latents)
        self.save_anim_latents_button.grid( row=3,
                                            column=0,
                                            sticky="NW")
        # Save encoded latents
        self.save_encod_latents_button = tk.Button(self.anim_container, text="Save real", command=self.save_encod_latents)
        self.save_encod_latents_button.grid( row=4,
                                            column=0,
                                            sticky="NW")
        # SAVE RESULT
        self.save_button = tk.Button(self.main_window, text="Export", command=self.export_image)
        self.save_button.grid(  row=self.animation_panel.grid_info()['row'], 
                                column=self.animation_panel.grid_info()["column"], 
                                sticky="NE")

        # WEBCAM
        self.webcam_panel.grid( row=1, 
                                column=1,
                                sticky="NW")
        self.stream_button = tk.Button(self.main_window, text="Webcam", relief="raised", command=self.toggle_webcam)
        self.stream_button.grid(    row=self.webcam_panel.grid_info()['row'], 
                                    column=self.webcam_panel.grid_info()['column'], 
                                    sticky="NW")

        # WEBCAM CAPTURE
        self.capture_panel = tk.Label(self.main_window)
        self.capture_panel.grid(    row=self.webcam_panel.grid_info()['row'], 
                                    column=self.webcam_panel.grid_info()['column']+1, 
                                    sticky="NW")
        self.capture_button = tk.Button(self.main_window, text="Capture", command=self.capture_webcam)
        self.capture_button.grid(   row= self.webcam_panel.grid_info()['row'], 
                                    column=self.webcam_panel.grid_info()['column'], 
                                    sticky="NE")

        # WEBCAM ENCODE  
        self.cencode_button = tk.Button(self.main_window, text="Encode", command=self.encode_capture)
        self.cencode_button.grid(   row= self.webcam_panel.grid_info()['row'], 
                                    column=self.webcam_panel.grid_info()['column']+1, 
                                    sticky="NW")


    # DIRECTIONS
    # Reset all directions
    def reset_all_directions(self):    
        for id in range(len(self.directions)):
            self.sliders_directions[id].set(0)                                                                         
        return None

    # Save directions
    def export_directions(self):
        title = self.save_directions.get() # "1.0",'end-1c'
        self.animation.save_directions(title)

    # Reset directions
    def directions_reset_pressed(self, id):
        self.sliders_directions[id].set(0)

    # Update directions
    def sliders_directions_changed(self, value, direction):
        print(direction, float(value))
        self.animation.set_directions(direction, float(value))

        # temp_list = list(self.directions[direction])
        # temp_list[3] = float(value)
        # self.directions[direction] = tuple(temp_list)
        # self.animation.set_directions(self.directions)

    # Set directions sliders values
    def sliders_directions_set(self):
        for direction in self.directions:
            direction_index = list(self.directions.keys()).index(direction)
            self.sliders_directions[direction_index].set(self.directions[direction][3])

    # ENCODED IMAGE
    def show_encoded(self):
        raw_img = self.animation.get_encoded_face()
        img = ImageTk.PhotoImage(raw_img)
        self.encoded_panel.configure(image=img)
        self.encoded_panel.img = img

    # ANIMATION
    def toggle_animation(self):
        if self.animation_button.config('relief')[-1] == 'sunken':
            self.animation.stop()
            self.animation_button.config(relief="raised")
        else:
            self.animation.start()
            self.animation_button.config(relief="sunken")

    # Change face
    def random_face(self):
        self.animation.change_face(random_face=True)

    def encoded_face(self):
        self.animation.change_face(random_face=False)

    # Jitter: Amplitude
    def jitter_amp_changed(self, event):
        amp_val = self.jitter_amp_val.get()
        self.animation.set_jitter_amp(amp_val)
            
    # Jitter: Speed
    def jitter_speed_changed(self, event):
        speed_val = self.jitter_speed_val.get()
        self.animation.set_jitter_speed(speed_val)

    # Save animation latents
    def save_anim_latents(self):
        self.animation.save_latents(encoded=False)

    # Save encoded latents
    def save_encod_latents(self):
        self.animation.save_latents(encoded=True)

    # SAVE RESULT
    def export_image(self):
        img = ImageTk.getimage(self.animation_panel.img)
        filename = "./exported_images/export_%i.png" % random.randrange(9999)
        img.save(filename)

    # WEBCAM
    def toggle_webcam(self):
        if self.stream_button.config('relief')[-1] == 'sunken':
            self.camera.stop()
            self.stream_button.config(relief="raised")
        else:
            self.camera.start()
            self.stream_button.config(relief="sunken")

    # WEBCAM CAPTURE
    def capture_webcam(self):
        capture_image = self.webcam_panel.img
        self.capture_panel.configure(image=capture_image)
        self.capture_panel.img = capture_image
        # Save captures image to disk
        ImageTk.getimage(self.webcam_panel.img).convert('RGB').save('./input.jpg')

    # WEBCAM ENCODE
    def encode_capture(self):
        self.animation.set_encoded_face("./input.jpg")
        self.show_encoded()

    # CLOSING WINDOW ROUTINE
    def __on_closing(self):
        self.animation.stop()
        self.camera.release()

        self.main_window.destroy()

    # RUN GUI
    def start(self):
        # INITIALIZATION
        self.show_encoded()
        self.toggle_animation()
        self.sliders_directions_set()

        self.main_window.mainloop()