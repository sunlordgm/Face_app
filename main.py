import cv2
import numpy as np
import tensorflow as tf
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.uix.filechooser import FileChooserIconView

class FaceApp(App):
    def build(self):
        # Carga Cascade y modelos TFLite
        self.fd = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')

        self.emo = tf.lite.Interpreter(model_path='models/emotions.tflite')
        self.emo.allocate_tensors()
        self.age = tf.lite.Interpreter(model_path='models/age_gender.tflite')
        self.age.allocate_tensors()

        return self.root

    def analizar_frame(self):
        cam = self.root.ids.cam
        # Obtiene imagen BGRA
        buf = cam.texture.pixels
        img = np.frombuffer(buf, np.uint8).reshape(cam.resolution[1], cam.resolution[0], 4)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.procesar(img)

    def cargar_imagen(self):
        chooser = FileChooserIconView()
        chooser.bind(on_submit=self._on_file)
        self.root.add_widget(chooser)

    def _on_file(self, chooser, selection, touch):
        img = cv2.imread(selection[0])
        self.root.remove_widget(chooser)
        self.procesar(img)

    def procesar(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.fd.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (48,48)) / 255.0
            inp = np.expand_dims(roi_resized, axis=(0,-1)).astype(np.float32)

            # Expresión
            i_idx = self.emo.get_input_details()[0]['index']
            o_idx = self.emo.get_output_details()[0]['index']
            self.emo.set_tensor(i_idx, inp)
            self.emo.invoke()
            emo_id = np.argmax(self.emo.get_tensor(o_idx))

            # Edad/Género
            i2 = self.age.get_input_details()[0]['index']
            o2 = self.age.get_output_details()[0]['index']
            self.age.set_tensor(i2, inp)
            self.age.invoke()
            ag = self.age.get_tensor(o2)[0]
            edad, genero = int(ag[0]), 'Mujer' if ag[1] > 0.5 else 'Hombre'

            etiqueta = f"{edad}a, {genero}, {['Enojado','Feliz','Triste','Neu'][emo_id]}"
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, etiqueta, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Muestra en Kivy
        buf2 = cv2.flip(img, 0).tobytes()
        tex = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        tex.blit_buffer(buf2, colorfmt='bgr', bufferfmt='ubyte')
        self.root.ids.cam.texture = tex

if __name__ == '__main__':
    FaceApp().run()