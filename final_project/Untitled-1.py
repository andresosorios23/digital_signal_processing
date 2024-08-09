from openpiv import tools, pyprocess, validation, filters, scaling
from typing import Any, Union, List, Optional
import numpy as np
from imageio.v3 import imread, imwrite as _imsave
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import imageio
import importlib_resources
import pathlib
from matplotlib.backends.backend_agg import FigureCanvasAgg


def display_vector_field_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    flags: np.ndarray,
    mask: np.ndarray,
    on_img: Optional[bool] = False,
    image_name: Optional[Union[pathlib.Path, str]] = None,
    window_size: Optional[int] = 32,
    scaling_factor: Optional[float] = 1.0,
    ax: Optional[Any] = None,
    width: Optional[float] = 0.0025,
    show_invalid: Optional[bool] = True,
    **kw,
):
    """Displays quiver plot of the data in five arrays: x,y,u,v and flags


    Parameters
    ----------
    filename :  string
        the absolute path of the text file

    on_img : Bool, optional
        if True, display the vector field on top of the image provided by
        image_name

    image_name : string, optional
        path to the image to plot the vector field onto when on_img is True

    window_size : int, optional
        when on_img is True, provide the interrogation window size to fit the
        background image to the vector field

    scaling_factor : float, optional
        when on_img is True, provide the scaling factor to scale the background
        image to the vector field

    show_invalid: bool, show or not the invalid vectors, default is True


    Key arguments   : (additional parameters, optional)
        *scale*: [None | float]
        *width*: [None | float]


    See also:
    ---------
    matplotlib.pyplot.quiver


    Examples
    --------
    --- only vector field
    >>> openpiv.tools.display_vector_field('./exp1_0000.txt',scale=100,
                                           width=0.0025)

    --- vector field on top of image
    >>> openpiv.tools.display_vector_field(Path('./exp1_0000.txt'), on_img=True,
                                          image_name=Path('exp1_001_a.bmp'),
                                          window_size=32, scaling_factor=70,
                                          scale=100, width=0.0025)

    """

    if isinstance(u, np.ma.MaskedArray):
        u = u.filled(0.0)
        v = v.filled(0.0)

    if mask is None:
        mask = np.zeros_like(u, dtype=int)

    if flags is None:
        flags = np.zeros_like(u, dtype=int)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if on_img is True:  # plot a background image
        im = imread(image_name)
        xmax = np.amax(x) + window_size / (2 * scaling_factor)
        ymax = np.amax(y) + window_size / (2 * scaling_factor)
        ax.imshow(im, cmap="Greys_r", extent=[0.0, xmax, 0.0, ymax])

    # first mask whatever has to be masked
    u[mask.astype(bool)] = 0.0
    v[mask.astype(bool)] = 0.0

    # now mark the valid/invalid vectors
    invalid = flags > 0  # mask.astype("bool")
    valid = ~invalid

    # visual conversion for the data on image
    # to be consistent with the image coordinate system

    # if on_img:
    #     y = y.max() - y
    #     v *= -1

    ax.quiver(x[valid], y[valid], u[valid], v[valid], color="b", width=width, **kw)

    if show_invalid and len(invalid) > 0:
        ax.quiver(
            x[invalid],
            y[invalid],
            u[invalid],
            v[invalid],
            color="r",
            width=width,
            **kw,
        )

    # if on_img is False:
    #     ax.invert_yaxis()

    ax.set_aspect(1.0)
    # fig.canvas.set_window_title('Vector field, '+str(np.count_nonzero(invalid))+' wrong vectors')

    return fig, ax


cap = cv2.VideoCapture("final_project/data/slow_car.mp4")
ret, frame = cap.read()
frames = []
while 1:

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(frame)


car_data = cv2.CascadeClassifier("final_project/data/cars.xml")


new_frames = []
current_y = 0
current_h = 0
for frame_original in frames:
    frame = frame_original.copy()
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    frame_dilated = cv2.dilate(frame_blur, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(frame_dilated, cv2.MORPH_CLOSE, kernel)
    cars = car_data.detectMultiScale(closing, 1.1, 1)
    if len(cars) == 0:
        frame[current_y + current_h :] = 0
        frame[: current_y - current_h - 50] = 0
        new_frames.append(frame)
        continue
    if len(cars) > 1:
        cars = cars[:1]
    for x, y, w, h in cars:
        frame[y + h :] = 0
        frame[: y - h - 50] = 0
        current_y = y
        current_h = h
        # frame[:, x + w + 500 :] = 0
        # frame[:, : x -100 ] = 0

    new_frames.append(frame)


# frame = frames[101].copy()
# frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
# frame_dilated = cv2.dilate(frame_blur, np.ones((3, 3)))
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
# closing = cv2.morphologyEx(frame_dilated, cv2.MORPH_CLOSE, kernel)
# cars = car_data.detectMultiScale(closing, 1.1, 1)
# cnt = 0
# for x, y, w, h in cars:
#     frame[y + h :] = 0
#     frame[:y - h -50] = 0
#     frame[:, x + w + 500 :] = 0
#     frame[:, : x -100 ] = 0

#     cnt += 1
# print(cnt, " cars found")
# plt.figure()
# plt.imshow(frame, cmap="gray")


fps = cap.get(cv2.CAP_PROP_FPS)
video = cv2.VideoWriter(
    "video.mp4",
    cv2.VideoWriter_fourcc("m", "p", "4", "v"),
    fps,
    (800, 800),
)
winsize = 50  # pixels, interrogation window size in frame A
searchsize = 60  # pixels, search area size in frame B
overlap = 30  # pixels, 50% overlap
dt = 1 / fps  # sec, time interval between the two frames
for i in range(0, len(new_frames) - 1):
    plt.figure()
    cv2.imwrite("frameA.png", frames[i])
    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        new_frames[i].astype(np.int32),
        new_frames[i + 1].astype(np.int32),
        window_size=winsize,
        overlap=overlap,
        dt=dt,
        search_area_size=searchsize,
        sig2noise_method="peak2peak",
    )

    x, y = pyprocess.get_coordinates(
        image_size=new_frames[50].shape,
        search_area_size=searchsize,
        overlap=overlap,
    )
    invalid_mask = validation.sig2noise_val(
        sig2noise,
        threshold=1.05,
    )

    u2, v2 = filters.replace_outliers(
        u0,
        v0,
        invalid_mask,
        method="localmean",
        max_iter=10,
        kernel_size=3,
    )

    # convert x,y to mm
    # convert u,v to mm/sec

    x, y, u3, v3 = scaling.uniform(
        x,
        y,
        u2,
        v2,
        scaling_factor=96.52,  # 96.52 pixels/millimeter
    )

    # 0,0 shall be bottom left, positive rotation rate is counterclockwise
    x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)
    angle = np.rad2deg(np.arctan2(v3, u3))
    u3[abs(angle) > 10] = 0
    v3[abs(angle) > 10] = 0
    speed = np.sqrt(u3**2 + v3**2)
    u3[speed < 0.2] = 0
    v3[speed < 0.2] = 0
    speed[speed < 0.6] = 0
    speed = 2100 / 1280 * speed
    fig, ax = plt.subplots(figsize=(8, 8))
    # tools.save("exp1_001.txt", x, y, u3, v3, invalid_mask)
    fig, ax = display_vector_field_from_arrays(
        x=x,
        y=y,
        u=u3,
        v=v3,
        flags=invalid_mask,
        mask=invalid_mask,
        ax=ax,
        scaling_factor=96.52,
        scale=50,  # scale defines here the arrow length
        width=0.0035,  # width is the thickness of the arrow
        on_img=True,  # overlay on the image
        image_name="frameA.png",
        show_invalid=False,
    )
    ax.title.set_text(f"Frame {i}: {speed[speed>0.1].mean():.2f} m/s")
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    frame = cv2.cvtColor(
        np.asarray(canvas.buffer_rgba(), dtype="uint8"), cv2.COLOR_RGB2BGR
    )
    # im = Image.fromarray(frame)
    # im.save("test.bmp")
    video.write(frame)
    plt.close()
    A = 1
video.release()
# plt.title(f"Frame {i}: {speed.mean():.2f} mm/s")
# plt.show()
