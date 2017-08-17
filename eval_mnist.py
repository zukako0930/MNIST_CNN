import sys

from keras.models import load_model
from mnist_helper import test_mnist

model_file = sys.argv[1]

model = load_model(model_file)

# 評価用 MNIST データセット取得
(x_test, y_test) = test_mnist()

# 評価
(loss, acc) = model.evaluate(x_test, y_test, verbose = 0)

print("loss = %f, accuracy = %f" % (loss, acc))