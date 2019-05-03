import os
import torch
import collections


class Coder:

    def __init__(self, alphabet_file):
        # All char in one line
        self.alphabet = open(alphabet_file, 'r').readlines()[0].replace('\n', '')
        self.dictionary = {}
        self.label_to_char = {}

        cnt = 1
        for ch in self.alphabet:
            self.dictionary[ch] = cnt
            self.label_to_char[cnt] = ch
            cnt += 1

    def encode(self, word_str):

        labels = []
        for ch in word_str:
            if ch in self.alphabet:
                labels.append(self.dictionary[ch])

        return labels

    def decode(self, labels):

        dec_str = ''
        for label in labels:
            if label in self.label_to_char:
                dec_str += self.label_to_char[label]

        return dec_str

class StrLabelConverter(object):

    def __init__(self, alphabet):
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
        print('------------------- alphabet -------------------')
        print('alphabet:', self.alphabet)
        print('------------------------------------------------')
    def encode(self, text, depth=0):
        """Support batch or single str."""
        if isinstance(text, str):
            for char in text:
                if self.alphabet.find(char) == -1:
                    print(char)
            text = [self.dict[char] for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)

        if depth:
            return text, len(text)
        #return (torch.IntTensor(text), torch.IntTensor(length))
        return (text, length)

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            t = t[:length]
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(
                    t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


if __name__ == '__main__':

    alpha_f = 'alpha.txt'
    coder = Coder(alpha_f)

    words = ['shits', 'bull', 'fXxk']

    for w in words:
        code = coder.encode(w)
        print('code:', code)
        word = coder.decode(code)
        print('word:', word)
