import streamlit as st
from openvino.runtime import Core
from PIL import Image
import cv2
from sklearn import preprocessing
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path
import librosa
import subprocess
from transformers import Wav2Vec2Processor

curr_cwd = Path.cwd()
if not Path.exists(curr_cwd / 'work_dir'):
    Path.mkdir(curr_cwd / 'work_dir')

uploaded_file = st.file_uploader("Choose a file")


def save_file(video):
    with open(curr_cwd / 'work_dir' / 'video.mp4', "wb") as f:
        f.write(video.getbuffer())


def preprocess_function_eval(speech_path):
    speech_array, sampling_rate = librosa.load(speech_path, sr=16000)
    result = processor(speech_array, sampling_rate=target_sampling_rate, max_length=50000, padding=True,
                       truncation=True, return_attention_mask=True)
    len_of_input_data = result['input_values'][0].shape[0]
    padded_array = np.pad(result['input_values'][0], ((0, 50000 - len_of_input_data)), constant_values=0)
    return padded_array


col1, col2, col3 = st.columns([3, 2, 2])

with col2:
    im_display_slot = st.empty()

if col2.button(label='Запустить анализ') and uploaded_file is not None:
    save_file(uploaded_file)
    DATA_DIR = curr_cwd / 'work_dir'
    video_path = curr_cwd / 'work_dir' / 'video.mp4'
    faces_path = curr_cwd / 'work_dir' / 'faces'
    audio_path = curr_cwd / 'work_dir' / 'audio'
    print(video_path)
    FRAMES_TO_SKIP = 1
    from facial_analysis import FacialImageProcessing

    #
    imgProcessing = FacialImageProcessing(False)

    if not Path.exists(faces_path):
        Path.mkdir(faces_path)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total_frames:', total_frames)

    frame_count = 0
    counter = 0

    for frame_count in tqdm(range(0, total_frames - 1, FRAMES_TO_SKIP)):
        ret, frame_bgr = cap.read()
        counter += FRAMES_TO_SKIP
        cap.set(cv2.CAP_PROP_POS_FRAMES, counter)
        if not ret:
            # cap.release()
            # break
            continue
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bounding_boxes, _ = imgProcessing.detect_faces(frame)
        if len(bounding_boxes) != 0:
            if len(bounding_boxes) > 1:
                bounding_boxes = bounding_boxes[:1]

            b = [int(bi) for bi in bounding_boxes[0]]
            x1, y1, x2, y2 = b[0:4]
            face_img = frame_bgr[y1:y2, x1:x2, :]

            outfile = os.path.join(faces_path, str(counter) + '.png')

            if np.prod(face_img.shape) == 0:
                print('Empty face ', b, ' found for ', outfile)
                continue
            cv2.imwrite(outfile, face_img)

    cap.release()
    ie = Core()
    # path to xml file with model converted to OpenVino's IR format. bin file should be in the same directory
    model = ie.read_model(model=curr_cwd / 'models'/'faces'/'enet_b0_8_FP16.xml')
    # adding extra output layer
    model.add_outputs(['652'])
    feature_extractor = ie.compile_model(model=model, device_name="CPU")
    input_layer = next(iter(feature_extractor.inputs))
    output_layer = feature_extractor.outputs[1]
    print(input_layer)
    print(output_layer)

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


    def torch_transforms(filepath):
        face_img = Image.open(filepath)
        img_tensor = test_transforms(face_img)
        img_arr = img_tensor.numpy()
        img_arr = np.expand_dims(img_arr, axis=0)
        return img_arr


    emotion_to_index = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}


    def get_features(data_dir):
        filename2features = {}

        video_scores = []
        X_isface = []
        for img_name in os.listdir(data_dir):
            filepath = os.path.join(data_dir, img_name)
            input_image = torch_transforms(filepath)
            X_isface.append('noface' not in img_name)

            if input_image.size:
                scores = feature_extractor(inputs=[input_image])[output_layer]
                scores = scores[0].squeeze()
                video_scores.append(scores)
        filename2features['video_name'] = (np.array(video_scores), np.array(X_isface))
        return filename2features


    filename2features = get_features(faces_path)


    def create_dataset(filename2features):
        x = []
        y = []
        has_faces = []
        ind = 0
        fn = 'video_name'

        features = filename2features[fn]
        total_features = None
        # print(len(features))
        if True:
            if len(features[0]) != 0:
                cur_features = features[0][features[-1] == 1]
            # print(prev,features.shape)
        else:
            cur_features = features[0]
        if len(cur_features) == 0:
            has_faces.append(0)
            total_features = np.zeros_like(feature)
        else:
            has_faces.append(1)
            mean_features = (np.mean(cur_features, axis=0))
            std_features = (np.std(cur_features, axis=0))
            max_features = (np.max(cur_features, axis=0))
            min_features = (np.min(cur_features, axis=0))

            # join several features together
            feature = np.concatenate((mean_features, std_features, min_features, max_features), axis=None)

            total_features = feature

        if total_features is not None:
            x.append(total_features)
        x = np.array(x)
        has_faces = np.array(has_faces)
        print(x.shape)
        return x, has_faces


    x_test, has_faces_test = create_dataset(filename2features)

    x_test_norm = preprocessing.normalize(x_test, norm='l2')
    # load the model from disk
    filename = 'linear_svc.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(x_test_norm)
    y_logits_faces = loaded_model.decision_function(x_test_norm)
    # print(y_pred)
    class_to_idx = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
    idx_to_class = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    st.write('Face model final result is: ', idx_to_class.get(y_pred[0]))

    # ----------------------------------------------------------------------------------------------------------------
    model_name_or_path = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    pooling_mode = "mean"
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
    target_sampling_rate = processor.feature_extractor.sampling_rate

    if not Path.exists(audio_path):
        Path.mkdir(audio_path)

    from openvino.runtime import Core

    subprocess.run(
        f'ffmpeg -y -i {video_path} -f wav -ab 192000 -ar 16000 -vn {audio_path / Path("audio").with_suffix(".wav")}',
        shell=True, check=True)
    # time.sleep(5)
    input_data = preprocess_function_eval(audio_path / 'audio.wav')

    ie = Core()
    classification_model_xml = curr_cwd / 'models'/'audio'/'wav2vec2.xml'  # <- insert here xml file where folder contains .bin weight file
    model = ie.read_model(model=classification_model_xml)
    model.reshape([1, 50000])
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    result_audio = compiled_model([np.expand_dims(input_data, 0)])[output_layer]
    st.write('Audio model final result is: ', idx_to_class.get(int(np.argmax(result_audio))))
    summed_logits = np.add(y_logits_faces, result_audio)
    emotion = np.argmax(summed_logits)

    st.write('Ensemble final result is: ', idx_to_class.get(int(emotion)))
#
#     y_pred
