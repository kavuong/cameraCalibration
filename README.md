# cameraCalibration

In calib.py, camera calibration in 2D and 3D was implemented by estimating planar projective transformations, generating
correspondences and estimating the projection matrix P for the entire calibration grid. That matrix P was then
decomposed into the camera calibration matrix K (normalized) and the matrix [R T] composing the rigid body motion of
the camera.

Each image point produces two linear equations in the unknowns of the projection matrix. There are eight image points
found, and each of those eight points corresponds to a set of corner coordinates given in 3D. Each of those 3D corners
are the extreme corners of the XZ and YZ planes. These image points are rewritten into their linear equations, which
are then rewritten into matrix form. By using linear least squares, I found Hxz and Hyz, 3x3 matrices that accurately
transformed 3D coordinates to their 2D image coordinate counterparts.

Those Hxz and Hyz matrices are used in generating correspondence between all 3D corners and their 2D counterparts.
First, every corner is stored with their 3D coordinates. Those 3D corners are then projected onto their respective
image planes using matrix multiplication with Hxz/Hyz. Then for each projection, the nearest real corner in 2D is found
and is associated with that 3D corner. A list of 3D corners and their corresponding 2D counterparts is returned.

To find the projection matrix P, a similar process as estimating the planar projective transformations is followed, but
it accounts for both the XZ and YZ planes rather than just one of them. The 6 corners picked for calibration cover all
the corners in both planes. The top left, top right, bottom left and bottom right corners are picked, as well as two
corners in the middle (one corner for each plane). Singular value decomposition is used to find the transpose of V.
After taking the transpose, the V array is returned, and the last column of V is the projective matrix P, but in 12x1
format rather than 3x4, so a reshape is taken.

The projection matrix P found in the previous part can be decomposed into its components K and [R T] using QR
decomposition. The projection matrix P is decomposed into an orthogonal matrix Q and R', which is then used to find K, R and T. During the process, K is normalized in order to make the last element of K equal to 1. T is appended to R in order to return K and [R T].

In epipolar.py, I compute the essential matrix E from two matrices representing the rigid body motions of two cameras.
First the relative motion matrix R and the relative transition matrix T are found. I then take the cross product of T and R's transpose, and then take the transpose of that result to return the essential matrix E.
