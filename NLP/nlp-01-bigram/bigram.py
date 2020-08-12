import os
import re
import math
import json
import random
from pprint import pprint

CORPUS_FILE = "corpus.txt"
TOKENS_FILE = "tokens.txt"
BIGRAM_FILE = "bigram.json"

def preprocess(text):
    # lower case text
    text = text.lower()
    # hapus angka dan simbol (selain a-z)
    text = re.sub(r'[^a-z/.]', ' ', text)
    # ganti titik sama flag akhir dan awal kalimat
    text = re.sub(r'[/.]', ' </s> <s> ', text)
    # tambah flag awal kalimat di awal teks
    text = "<s> " + text
    return text

def tokenise(text):
    # misahin perkata
    tokens = text.split()
    # Remove trailing sentence-start flag at the very end of tokens
    if tokens[-1] == "<s>":
        tokens.pop()
    return tokens

def build_bigram(source_file=CORPUS_FILE):
    # Read from corpus
    with open(source_file, encoding="utf8") as f:
        corpus = f.read()

    # Tokenise the corpus
    processed_corpus = preprocess(corpus)
    tokens = tokenise(processed_corpus)

    # Get all the unique tokens
    unique_tokens = set(tokens)
    
    # Save list of tokens to txt
    with open(TOKENS_FILE, mode='w') as f:
        for token in unique_tokens:
            f.write("{}\n".format(token))

    # Initialize unigram and bigram
    unigram = {}
    bigram = {}
    for word_row in unique_tokens:
        unigram[word_row] = 0
        bigram[word_row] = {}
        for word_column in unique_tokens:
            bigram[word_row][word_column] = 0

    # Build the bigram and unigram from word sequences in tokens
    unigram[ tokens[0] ] += 1
    for i in range(1, len(tokens)):
        unigram[ tokens[i] ] += 1
        bigram[ tokens[i-1] ][ tokens[i] ] += 1

    # Laplace (add-one) smoothing
    for word_row in bigram:
        # Exclude sentence-end flag because it's always followed by sentence-start flag
        if word_row == "</s>":
            continue
        unigram[word_row] += len(unigram) if word_row != "<s>" else len(unigram) - 2
        for word_column in bigram[word_row]:
            bigram[word_row][word_column] += 1

    # Normalize the bigram with unigram
    for word_row in bigram:
        for word_column in bigram[word_row]:
            bigram[word_row][word_column] /= unigram[word_row]
    
    # Special case handling for flags
    bigram["<s>"]["</s>"] = 0.0
    bigram["<s>"]["<s>"] = 0.0
    bigram["</s>"]["<s>"] = 1.0

    # Save bigram to json file
    with open(BIGRAM_FILE, mode='w') as f:
        json.dump(bigram, f)
    
    return bigram

def predict_word(bigram, preceding_words):
    if not preceding_words:
        return 
    # Preprocess and tokenise the preceding words
    preceding_tokens = tokenise(preprocess(preceding_words))
    if preceding_tokens[-1] not in bigram:
        return
    # Return the possibilities of each word in model as the next word (sorted descending)
    next_words = bigram[ preceding_tokens[-1] ]
    return sorted(random.sample( list(next_words.items()), len(next_words) ), key=lambda x: x[1], reverse=True)

def sentence_likelihood(bigram, sentence):
    if not sentence:
        return
    # Preprocess and tokenise the sentence
    sentence_tokens = tokenise(preprocess(sentence))
    if sentence_tokens[-1] != "</s>":
        sentence_tokens.append("</s>")
    if any(token not in bigram for token in sentence_tokens):
        return
    log_sum = 0.0
    for i in range(1, len(sentence_tokens)):
        prob = bigram[ sentence_tokens[i-1] ][ sentence_tokens[i] ]
        prob_log = math.log(prob)
        print("p ({}|{}) = {} %".format( sentence_tokens[i], sentence_tokens[i-1], round(prob * 100, 5) ))
        
        if sentence_tokens[i-1] == "<s>":
            start_prob_log = prob_log
        if sentence_tokens[i] == "</s>":
            end_prob_log = prob_log
        log_sum += prob_log
    
    partial_sentence_prob = math.pow(math.e, log_sum - start_prob_log - end_prob_log)
    whole_sentence_prob = math.pow(math.e, log_sum)
    return (partial_sentence_prob, whole_sentence_prob)

def generate_sentences(bigram, n_word):
    if n_word <= 0 or len(bigram) < 1:
        return "<Error: Kata/bigram tidak valid>"
    # First word
    word = random.choice(list(bigram))
    sentence = word + " "
    for _ in range(n_word-1):
        variat = random.random()
        for next_word in bigram[word]:
            variat -= bigram[word][next_word]
            if variat <= 0:
                word = next_word
                break
        sentence += word + " "
    return sentence

# Main program
if __name__ == '__main__':
    # Load bigram data if already exists; otherwise create new
    if os.path.isfile(BIGRAM_FILE):
        print("Model bigram ditemukan pada \"{}\".".format(BIGRAM_FILE))
        with open(BIGRAM_FILE) as f:
            bigram = json.load(f)
        print("\"{}\" telah di-load!".format(BIGRAM_FILE))
    else:
        print("Model bigram tidak ditemukan. Buat model baru pada menu")
        bigram = {}

    # Main menu
    menu_choice = ""
    while menu_choice != "0":
        print("\nBigram Bahasa Indonesia")
        print("Jumlah token: {} token\n".format(len(bigram)))
        print("1. Prediksi kemunculan kata")
        print("2. Hitung peluang kemunculan kalimat")
        print("3. Buat rangkaian kata")
        print("4. Bangun model bigram")
        print("\n0. Keluar")
        menu_choice = input("\n>> ")
        
        if menu_choice == "1":
            preceding_words = input("\nDeretan kata: ")
            likely_words = predict_word(bigram, preceding_words)
            if likely_words:
                print("Kata-kata yang mungkin:")
                for i, word in enumerate(likely_words[:5], start=1):
                    print("{}. ({} %) {} {}".format( i, round(word[1] * 100, 5), preceding_words, word[0] ))
            else:
                print("Model tidak dapat menemukan kata yang sesuai.")
            input("\nTekan [enter] untuk kembali ke menu")
        
        elif menu_choice == "2":
            sentence = input("\nKalimat: ")
            likelihood = sentence_likelihood(bigram, sentence)
            if likelihood:
                print("\n---Probabilitas kalimat muncul---")
                print("Kalimat sebagian: {} %".format( round(likelihood[0] * 100, 10) ))
                print("Kalimat utuh    : {} %".format( round(likelihood[1] * 100, 10) ))
            else:
                print("Probabilitas tidak dapat dihitung karena ada kata yang tidak dikenali.")
            input("\nTekan [enter] untuk kembali ke menu")
        
        elif menu_choice == "3":
            n_word = int(input("\nJumlah kata: "))
            print("\n{}".format( generate_sentences(bigram, n_word) ))
            input("\nTekan [enter] untuk kembali ke menu")

        elif menu_choice == "4":
            source_file = input("\nFile korpus (kosongkan untuk default): ")
            if source_file == "":
                source_file = CORPUS_FILE
            bigram = build_bigram(source_file)
            print("Model bigram telah dibuat ulang dan disimpan pada \"{}\".".format(BIGRAM_FILE))
            print("Daftar token disimpan pada \"{}\".".format(TOKENS_FILE))
            input("\nTekan [enter] untuk kembali ke menu")