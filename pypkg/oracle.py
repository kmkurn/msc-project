from collections import namedtuple


class Oracle(namedtuple('Oracle', 'parsed, postags, words, lowercased, unkified, actions')):
    __slots__ = ()  # instance will take up the same memory as a regular tuple

    def __str__(self):
        temp = [self.parsed, self.postags, self.words, self.lowercased, self.unkified]
        temp.extend(self.actions)
        return '\n'.join(temp)


class GenOracle(namedtuple('GenOracle', 'parsed, words, unkified, actions')):
    __slots__ = ()

    def __str__(self):
        temp = [self.parsed, self.words, self.unkified]
        temp.extend(self.actions)
        return '\n'.join(temp)


def make_oracle_iter(class_, num_preamble):
    def oracle_iter(lines):
        count = 0
        buff, actions = [], []
        for line in lines:
            line = line.strip()
            if not line:
                yield class_(*buff, actions=actions)
                count = 0
                buff, actions = [], []
                continue
            if count < num_preamble:
                buff.append(line)
            else:
                actions.append(line)
            count += 1
        if buff:
            yield class_(*buff, actions=actions)
    return oracle_iter


oracle_iter = make_oracle_iter(Oracle, 5)
gen_oracle_iter = make_oracle_iter(GenOracle, 3)
