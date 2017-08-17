import sys

from keras.models import save_model, load_model
from mnist_helper import train_mnist

epoch = int(sys.argv[1])
mini_batch = int(sys.argv[2])

model_file = sys.argv[3]
model_dest_file = sys.argv[4]

# モデルの読み込み
model = load_model(model_file)

# 学習用 MNIST データセット取得
(x_train, y_train) = train_mnist()

# 学習
model.fit(x_train, y_train, nb_epoch = epoch, batch_size = mini_batch)

# 学習後のモデルを保存
save_model(model, model_dest_file)