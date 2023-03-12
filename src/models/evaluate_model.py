import click
from tqdm import tqdm
from models.read_data import read_split_dataset
from models.utils import build_saved_model
from features.nlp.tokenizer import CustomSpacyTokenizer
from models.predict_model import predict
from models.metrics import bleu

@click.command()
@click.option(
    "--model_filename",
    default="",
    help="Filename (without extension) of the model config and weights to load",
)
@click.option("--mode", default="test", help="Evaluation mode: 'val' or 'test', default is 'test'")
def main(model_filename, mode):
    model = build_saved_model(model_filename)
    tokenizer = CustomSpacyTokenizer.from_json()
    dataset = read_split_dataset(mode, model.img_shape, model.caption_length, batch_size=5)

    print(f"MODEL {model_filename}")
    for beam_width in 1, 3, 5:
        print(f"BEAM WIDTH: {beam_width}")
        total_bleu = [0., 0.]
        n = 0

        for (image, captions), _ in tqdm(dataset):
            n += 1
            prediction = predict(model, image[0], tokenizer, beam_width)
            clean_prediction = tokenizer.clean_text(prediction)
            
            captions = [tokenizer.sequence_to_text(captions[i].numpy()) for i in range(5)]
            clean_captions = [tokenizer.clean_text(caption) for caption in captions]

            example_bleu = bleu(clean_captions, clean_prediction)
            total_bleu[0] += example_bleu[0]
            total_bleu[1] += example_bleu[1]

        total_bleu[0] /= n
        total_bleu[1] /= n

        print(f"BLEU-1:{total_bleu[0]}\nBLEU-4:{total_bleu[1]}")


if __name__ == "__main__":
    main()