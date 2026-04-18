run: pip install numpy matplotlib

then run: python drone_vision_gui.py

Functions breakdown:

generate_sphere_points(n) generates 3000 evenly spaced points (we could increase the value of this var for a greater accuracy) in an imaginary sphere that is created around the drone, which acts as the perfect, theoretical camera coverage.

get_camera_vectors(yaw_deg, pitch_deg) gives each camera a yaw and pitch, which allows us to create the pyramids that show and calculate the space covered by each 55x70 camera, through 3d vectors.

evaluate_coverage(...) takes that theoretical sphere, allowing us to see how many of the 3000 spatial points come in contact with any of the pyramids created by the 55x70 cameras, allowing us to determine the percent covered and the amount not covered. This also allows us to visualize the locations of the blind spots by calculating the points the pyramid does not cover. 

_build_ui(self) adds interactions and builds the tkinter GUI.

_compute_worker(self) runs the simulation/math when you click the run button. It tests every 5-degree yaw and pitch for each 55x70 camera using dot products, and optimizes it using a Greedy Set Cover (picks the camera that covers the most currently uncovered spatial points and does this 16x). Then, it uses simulated annealing to change the angles slightly for each camera, acting as a further optimization by accepting bad and good moves with a probability that decays exponentially.

_update_3d(self, a) maps the optimization, cameras, drone body shape, propellers, blind spots, and field of views in a 3D field to make it easier to understand and find issues.  
