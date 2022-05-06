# すでに学習済みの場合は削除する
if [ -e ./models/research/saved_model_01 ]; then
    # 存在する場合
	rm -rf ./models/research/saved_model_01
fi

# 保存先のデータの保管
dir="./models/research/object_detection_tools_gelehrte/data/"
train_dir=$dir"train/"
val_dir=$dir"val/"

# 保存済みのデータの削除
rm $train_dir*.tfrecord
rm $val_dir*.tfrecord


for file in `\find ./image/tags/languageecho-TFRecords-export -maxdepth 1 -type f -name "*.tfrecord"` ; do
    if [[ $(($RANDOM % 10)) -lt 7 ]] ; then
	    cp $file $train_dir
	else
		cp $file $val_dir
	fi
done

cd ./models/research/object_detection_tools_gelehrte/data
./change_tfrecord_filename.sh

cd ../../../../