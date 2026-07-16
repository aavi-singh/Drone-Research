Setup: 16 cameras on a 300 mm-radius body. The coverage was evaluated by seeing overlap with 50k points at 2m from the body’s surface. 
FOV used was 80x65 
Optimization technique: gradient descent for yaw, pitch, and roll for all cameras simultaneously. 48-dimensions used for the gradient descent. ⛛f(y1, p1, r1, … , y16, p16, r16). Each derivative is an approximation as h = 0.003 deg instead of approaching 0. 
Propellers were not considered. 
Most arrangements after optimization had near-perfect coverage, so the best method should consider stereo overlap and eventually propellers and location of the circuits connecting the cameras. 
