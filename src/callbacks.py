import os
import tensorflow as tf
from data.labels import drawTF, drawHmTF


class DrawImageCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        logdir,
        tfdataset,
        drawSteps=1,
        thres=0.2,
        writerName="imager",
    ):
        super(DrawImageCallback, self).__init__()

        self.tbcb = tf.summary.create_file_writer(os.path.join(logdir, writerName))
        self.writerName = writerName
        self.step_number = 0
        self.drawSteps = 0
        self.thres = thres
        self.tfdataset = tfdataset


    def on_epoch_end(self, epoch, logs=None):
        """Draw images at the end of an epoche

        Args:
            epoch ([type]): [description]
            logs ([type], optional): [description]. Defaults to None.
        """

        if self.step % self.drawSteps == 0:
            
            x, ytrue = None, None

            for (x, ytrue) in self.tfdataset:
                ypred = self.model.predict(x)
                break

            imgsTrue = drawTF(x, ytrue, thres=self.thres, normalizedImage=True)
            imgsPred = drawTF(x, ypred, thres=self.thres, normalizedImage=True)

            hmTrue = drawHmTF(ytrue)
            hmPred = drawHmTF(ypred)

            with self.tbcb.as_default():

                tf.summary.image(
                    "ImagesPred", imgsPred, max_outputs=25, step=self.step_number
                )
                tf.summary.image(
                    "ImagesTrue", imgsTrue, max_outputs=25, step=self.step_number
                )

                tf.summary.image(
                    "hmPred", hmTrue, max_outputs=25, step=self.step_number
                )
                tf.summary.image(
                    "hmTrue", hmPred, max_outputs=25, step=self.step_number
                )


        self.step_number += 1