# すでに学習済みの場合は削除する
if [ -e ./models/research/saved_model_01-FF14-letters ]; then
    # 存在する場合
	rm -rf ./models/research/saved_model_01-FF14-letters
fi

# 保存先のデータの保管
dir="./models/research/object_detection_tools_gelehrte/data-FF14-letters/"
train_dir=$dir"train/"
val_dir=$dir"val/"

# 公開用フォルダ
open_dir="../languageecho-tfrecord/FF14-letters/"

# 保存済みのデータの削除
rm $train_dir*.tfrecord
rm $val_dir*.tfrecord

# 公開用フォルダのtfrecordの削除
rm $open_dir*.tfrecord

dir_t="./image-FF14-letters/tags/languageecho-FF14-letters-TFRecords-export/"
cat FF14-letters_180_train.txt | while read line
do
	col1=`echo ${line} | cut -d ',' -f 1`
	file=$dir_t$col1
	echo "train--"
	echo $file
	cp $file $train_dir
done

cat FF14-letters_180_val.txt | while read line
do
	col1=`echo ${line} | cut -d ',' -f 1`
	file=$dir_t$col1
	echo "val--"
	echo $file
	cp $file $val_dir
done

for file in `\find ./image-FF14-letters/tags/languageecho-FF14-letters-TFRecords-export -maxdepth 1 -type f -name "*.tfrecord"` ; do
	cp $file $open_dir
	if [[ $(($RANDOM % 10)) -lt 7 ]] ; then
	    cp $file $train_dir
	else
		cp $file $val_dir
	fi
done

cd ./models/research/object_detection_tools_gelehrte/data-FF14-letters
./change_tfrecord_filename.sh

cd ../../../../