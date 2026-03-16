import argparse
from pathlib import Path
import pandas as pd
import joblib

from audio_source import AudioFileSource
from extract_features import Extract_Features
from train_model import Train_Model
from predict import Predict

def build_dataset_from_list(audio_list_csv, output_csv, sample_rate=16000):
    """
    audio_list_csv: CSV with columns path,label
    """
    afs = AudioFileSource(sample_rate=sample_rate)
    ext = Extract_Features(sample_rate=sample_rate)
    df = pd.read_csv(audio_list_csv)
    rows = []
    for _, row in df.iterrows():
        path = row['path']
        label = row['label']
        audio = afs.load(path)
        if audio is None:
            print(f"Skipping {path}")
            continue
        feats = ext.extract(audio)
        feats['label'] = label
        rows.append(feats)
    if not rows:
        raise RuntimeError("No valid audio files processed.")
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote dataset with {len(out_df)} rows to {output_csv}")
    return output_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['build', 'train', 'predict'], required=True)
    parser.add_argument('--audio_list', help='CSV file with columns path,label for building dataset')
    parser.add_argument('--data_csv', help='CSV of extracted features for training')
    parser.add_argument('--model_out', default='trained_pipeline.pkl', help='Path to save trained model')
    parser.add_argument('--predict_file', help='Audio file path to predict')
    args = parser.parse_args()

    if args.mode == 'build':
        if not args.audio_list or not args.data_csv:
            print('Provide --audio_list and --data_csv')
            return
        build_dataset_from_list(args.audio_list, args.data_csv)

    elif args.mode == 'train':
        if not args.data_csv:
            print('Provide --data_csv for training')
            return
        trainer = Train_Model(args.data_csv)
        X,y = trainer.load_dataset()
        trainer.baseline_cv(X,y)
        results = trainer.tune_and_train(X,y)
        if trainer.best_estimator_ is not None:
            joblib.dump(trainer.best_estimator_, args.model_out)
            print('Saved model to', args.model_out)
        else:
            print('No trained estimator to save.')

    elif args.mode == 'predict':
        if not args.predict_file:
            print('Provide --predict_file')
            return
        predictor = Predict(args.model_out)
        afs = AudioFileSource()
        ext = Extract_Features()
        audio = afs.load(args.predict_file)
        feats = ext.extract(audio)
        # ensure order matches predictor.feature_columns
        values = [feats.get(f) for f in predictor.feature_columns]
        label, conf = predictor.predict(features=values)
        print('Prediction:', label, 'Confidence:', conf)

if __name__ == '__main__':
    main()
