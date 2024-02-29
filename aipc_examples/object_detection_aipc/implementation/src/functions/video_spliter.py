import cv2
import os
import json
#from tqdm import trange


class Video_Spliter:
    """
    Saves the frames of a video and rebuilds the file
    """

    def split_video(self, video_file_path: str, output_folder_root: str) -> None:
        """
        Creates the processing of saving a video into frames

        Args:
          video_file_path: The path to the video file.
          output_folder_root: The path to the folder where the frames should be saved
        """

        file_base_name = os.path.basename(video_file_path)
        images_output_folder = os.path.join(
            output_folder_root, file_base_name, "frames"
        )
        self.save_frames(video_file_path, images_output_folder)
        fps = self.get_fps(video_file_path)
        height, width, n_frames, codec = self.get_video_shape(video_file_path)
        video_description = {
            "file_name": file_base_name,
            "height": height,
            "width": width,
            "n_frames": n_frames,
            "codec": codec,
            "fps": int(fps),
        }
        with open(
            os.path.join(output_folder_root, file_base_name, "video_description.json"),
            "w",
        ) as file_descriptor:
            json.dump(video_description, file_descriptor, indent=4)

    def frames_to_video(self, root_folder_path, use_default_codec = True):
        """
        Creates the processing of saving frames into a video file

        Args:
          root_folder_path: The path to the folder where has a frames folder and a video_description.json
          use_default_codec: Default False. Use True in case opencv was installed with pip
        """
        with open(
            os.path.join(root_folder_path, "video_description.json"), "r"
        ) as file_vd:
            video_description = json.load(file_vd)

            self.create_video(
                os.path.join(root_folder_path, "frames"),
                os.path.join(root_folder_path, video_description["file_name"]),
                video_description,
                use_default_codec,
            )

    def save_frames(self, video_file: str, output_folder: str):
        """Saves all the frames of a video file to a folder.

        Args:
          video_file: The path to the video file.
          output_folder: The path to the folder where the frames should be saved.
        """

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_file)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            filename = os.path.join(output_folder, f"frame_{i:08d}.jpg")
            cv2.imwrite(filename, frame)
            i += 1

        cap.release()

    def get_fps(self, video_file: str):
        """Gets the fps of a video file using OpenCV.

        Args:
          video_file: The path to the video file.

        Returns:
          The fps of the video file as a floating point number.
        """

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        return fps

    def get_video_shape(self, video_file: str):
        """Gets the shape of a video using OpenCV.

        Args:
          video_file: The path to the video file.

        Returns:
          A tuple of the height, width, number of frames of the video and codec.
        """

        cap = cv2.VideoCapture(video_file)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = (
            chr(h & 0xFF)
            + chr((h >> 8) & 0xFF)
            + chr((h >> 16) & 0xFF)
            + chr((h >> 24) & 0xFF)
        )
        cap.release()

        return height, width, n_frames, codec

    def create_video(self, image_folder: str, output_file: str, video_descriptor: dict, use_default_codec=False):
        """Creates a video from a folder of image frames.

        Args:
          image_folder: The path to the folder containing the image frames.
          output_file: The path to the output video file.
        """
        # read first frame
        filepath = os.path.join(image_folder, f"frame_00000000.jpg")
        frame = cv2.imread(filepath)

        if not use_default_codec:
            video_writer = cv2.VideoWriter(
                output_file,
                cv2.VideoWriter_fourcc(*video_descriptor["codec"]),
                int(video_descriptor["fps"]),
                (frame.shape[1], frame.shape[0]),
            )
        else:
            video_writer = cv2.VideoWriter(
                output_file+".avi",
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                int(video_descriptor["fps"]),
                (frame.shape[1], frame.shape[0]),
            )

        for i in range(int(video_descriptor["n_frames"])):
            filepath = os.path.join(image_folder, f"frame_{i:08d}.jpg")
            frame = cv2.imread(filepath)
            video_writer.write(frame)

        video_writer.release()


if __name__ == "__main__":
    video_file = "/home/vbezerra/2023-04-13 10-56-12.mkv"
    output_folder = "./data/input.mp4/"

    v = Video_Spliter()
    v.split_video(video_file, "./data/")

    v.frames_to_video("/home/vbezerra/Documents/split-video/data/2023-04-13 10-56-12.mkv/")
