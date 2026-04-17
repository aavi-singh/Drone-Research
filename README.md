run: pip install numpy matplotlib

then run: python drone_vision_gui.py

Functions breakdown:

generate_sphere_points(n) generates 3000 evenly spaced points (we could increase the value of this var for a greater accuracy) in an imaginary sphere that is created around the drone, which acts as the perfect, theoretical camera coverage.

get_camera_vectors(yaw_deg, pitch_deg) gives each camera a yaw and pitch, which allows us to create the pyramids that show and calculate the space covered by each 55x70 camera, through 3d vectors.

evaluate_coverage(...) takes that theoretical sphere, sees how much of the total pyramids' volumes cover the volume of that sphere, allowing us to determine the percent covered and the amount not covered. This also allows us to visualize the locations of the blind spots by calculating the points the pyramid does not cover. 

