# Recurrent Neural Network Grammars for Low-Resource Languages

This repository contains an implementation for my final project *Exploring Recurrent Neural Network Grammars for Parsing Low-Resource Languages*, MSc in Artificial Intelligence, University of Edinburgh. The implementation extended the [original recurrent neural network grammars' by Dyer et al. (2016)](https://github.com/clab/rnng/). Before put under this repository, their code was forked and modified under [this repository](http://github.com/kmkurn/rnng). Any changes from the original code are thus can be viewed in that repository's commit history. This `README` file is also largely based on theirs.

## Prerequisites

 * Python 2.7
 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries
 * [Eigen](http://eigen.tuxfamily.org) (latest development release)
 * [CMake](http://www.cmake.org/)
 * [EVALB](http://nlp.cs.nyu.edu/evalb/) (latest version; please put the EVALB folder under `src` directory)
 * [brown-cluster](https://github.com/percyliang/brown-cluster) (version 1.3)

## Python package installation

The `pypkg` package needs to be installed. Run `pip -e .` from the project root directory.

## Build instructions

    mkdir src/build
    cd src/build
    cmake -DEIGEN3_INCLUDE_DIR=/path/to/eigen ..
    make

By default, pretrained word embeddings feature for the generative model is enabled. To disable it, pass `-DDEFINE_ENABLE_PRETRAINED=OFF` option to `cmake`. You can also pass `-j [number of threads]` option to enable multithreading for your compilation.

## Input file format

See `sample_input_english.txt` (English Penn Treebank) and `sample_input_indonesian.txt` (Indonesian Treebank)

## Oracles

The oracle-generation scripts convert a bracketed parse tree into a sequence of actions. The script also converts singletons in the training set and unknown words in the dev and test set into the appropriate `UNK` tokens.

### Obtaining the oracle for the discriminative model

For English

    python src/get_oracle.py [training file] [training file] > train.oracle
    python src/get_oracle.py [training file] [dev file] > dev.oracle
    python src/get_oracle.py [training file] [test file] > test.oracle

For Indonesian

    python src/get_oracle_id.py [training file] [training file] > train.oracle
    python src/get_oracle_id.py [training file] [dev file] > dev.oracle
    python src/get_oracle_id.py [training file] [test file] > test.oracle

### Obtaining the oracle for the generative model

For English

    python src/get_oracle_gen.py [training file] [training file] > train_gen.oracle
    python src/get_oracle_gen.py [training file] [dev file] > dev_gen.oracle
    python src/get_oracle_gen.py [training file] [test file] > test_gen.oracle

For Indonesian

    python src/get_oracle_gen_id.py [training file] [training file] > train_gen.oracle
    python src/get_oracle_gen_id.py [training file] [dev file] > dev_gen.oracle
    python src/get_oracle_gen_id.py [training file] [test file] > test_gen.oracle

## Discriminative model

The discriminative variant of the RNNG is used as a proposal distribution for decoding the generative model (although it can also be used for decoding on its own).

### Training the discriminative model

To train the discriminative model **without** character embeddings, run the command below

    build/nt-parser/nt-parser --cnn-mem 1700 -x -T [training oracle file] -d [dev oracle file] -C [original dev file (PTB bracketed format, see sample_input_english.txt)] -P -t --pretrained_dim [dimension of pretrained word embeddings] -w [pretrained word embeddings file] --lstm_input_dim 128 --hidden_dim 128 -D [dropout rate] --model_dir [directory to save the model] > log.txt

To use character embeddings, run the command using `build/nt-parser/nt-parser-char` binary instead, and add additional options `--char_embeddings_model addition --separate_unk_embeddings`. Both commands **MUST** be run from the `src` directory.

If pretrained word embeddings are not used, remove the `--pretrained_dim` and `-w` options.

This command will train the discriminative model with early stopping: every 25 parameter updates, the model is evaluated on the dev file, and the training will be stopped if there is no improvement after 10 evaluations. The training log is printed to `log.txt` (including information of the filename which the model is saved to, which is used for decoding with the `-m` option below).

Run the command with `-h` option to see all the available options.

### Decoding with discriminative model

To decode using the discriminative model **without** character embeddings, run the command below

    build/nt-parser/nt-parser --cnn-mem 1700 -x -T [training_oracle_file] -p [test_oracle_file] -C [original_test_file (PTB bracketed format, see sample_input_english.txt)] -P --pretrained_dim [dimension of pretrained word embeddings] -w [pretrained word embeddings file] --lstm_input_dim 128 --hidden_dim 128 -m [path to model parameter file] > output.txt

To use character embeddings, run the command using `build/nt-parser/nt-parser-char` binary instead, and add additional options `--char_embeddings_model addition --separate_unk_embeddings`. Both commands **MUST** be run from the `src` directory.

The output will be stored in `/tmp/parse/parser_test_eval.xxxx.txt` and the parser will output F1 score calculated with `EVALB` with options specified in `COLLINS.prm` file. The parameter file (following the `-m` in the command above) can be obtained from `log.txt`.

If training was done using pretrained word embeddings (by specifying the `-w` and `--pretrained_dim` options) or POS tags (`-P` option), then decoding must also use that same options.

Run the command with `-h` option to see all the available options.

## Generative model

In (Dyer et al., 2016), the generative model achieved state of the art results, and decoding is done using sampled trees from the trained discriminative model.

### Training word clusters

Word clusters are trained with Brown clustering algorithm, whose implementation is available [here](https://github.com/percyliang/brown-cluster). You should install the implementation on your machine. To train the clusters, first run the command to get the unkified words from an oracle file

    python scripts/get_unkified_from_oracle.py [training_oracle_file] > unkified.txt

Then, do Brown clustering on the output file

    python scripts/do_brown_cluster.py --outdir cluster-dir unkified.txt

The clusters are available in `cluster-dir/paths` file. Invoke any of the scripts above with `-h` or `--help` flag to see all the available options.

### Training the generative model

To train the generative model **without** character embeddings, run the command below

    build/nt-parser/nt-parser-gen -x -T [training oracle generative] -d [dev oracle generative] -t --clusters [path to clusters file, or clusters-train-berk.txt for PTB full dataset] --pretrained_dim [dimension of pretrained word embeddings] -w [pretrained word embeddings file] --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -D [dropout rate] --model_dir [directory to save the model] > log_gen.txt

To use character embeddings, run the command using `build/nt-parser/nt-parser-gen-char` binary instead, and add additional options `--char_embeddings_model addition --separate_unk_embeddings`. Both commands **MUST** be run from the `src` directory.

If pretrained word embeddings are not used, remove the `--pretrained_dim` and `-w` options.

This command will train the generative model with early stopping; every 25 parameter updates, the model is evaluated on the dev file, and the training will be stopped if there is no improvement after 10 evaluations. The training log is printed to `log_gen.txt`, including information on where the parameters of the model is saved to, which is used for decoding later.

### Decoding with the generative model

Decoding with the generative model requires sample trees from the trained discriminative model. All the commands below must be run from the `src` directory.

#### Sampling trees from the discriminative model

To sample trees from the discriminative model **without** character embeddings, run the command below

    build/nt-parser/nt-parser --cnn-mem 1700 -x -T [training oracle file] -p [test oracle file] -C [original test file (PTB bracketed format, see sample_input_english.txt)] -P --pretrained_dim [dimension of pretrained word embeddings] -w [pretrained word embeddings file] --lstm_input_dim 128 --hidden_dim 128 -m [parameter file of the trained discriminative model] --alpha [flattening coefficient] -s 100 > test-samples.props

To use character embeddings, run the command using `build/nt-parser/nt-parser-char` binary instead, and add additional options `--char_embeddings_model addition --separate_unk_embeddings`. Both commands **MUST** be run from the `src` directory.

If pretrained word embeddings are not used, remove the `--pretrained_dim` and `-w` options.

Important parameters:

 * s = # of samples (all reported results used 100)
 * alpha = flattening coefficient (value not exceeding 1 is sensible; may be better to tune this on dev set)

#### Prepare samples for likelihood evaluation

    utils/cut-corpus.pl 3 test-samples.props > test-samples.trees

#### Evaluate joint likelihood under generative model

To evaluate the samples using the generative model **without** character embeddings, run the command below

    build/nt-parser/nt-parser-gen -x -T [training oracle generative] --clusters [path to clusters file, or clusters-train-berk.txt for PTB full dataset] --pretrained_dim [dimension of pretrained word embeddings] -w [pretrained word embeddings file] --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -p test-samples.trees -m [parameters file from the trained generative model, see log_gen.txt] > test-samples.likelihoods

To use character embeddings, run the command using `build/nt-parser/nt-parser-gen-char` binary instead, and add additional options `--char_embeddings_model addition --separate_unk_embeddings`.

If pretrained word embeddings are not used, remove the `--pretrained_dim` and `-w` options.

#### Estimate marginal likelihood (final step to get language modeling ppl)

    utils/is-estimate-marginal-llh.pl 2416 100 test-samples.props test-samples.likelihoods > llh.txt 2> rescored.trees

 * 100 = # of samples
 * 2416 = # of sentences in test set
 * `rescored.trees` will contain the reranked samples

The file `llh.txt` would contain the final language modeling perplexity after marginalization (see the last lines of the file)

#### Compute generative model parsing accuracy (final step to get parsing accuracy from the generative model)

    utils/add-fake-preterms-for-eval.pl rescored.trees > rescored.preterm.trees
    utils/replace-unks-in-trees.pl [Discriminative oracle for the test file] rescored.preterm.trees > hyp.trees
    utils/remove_dev_unk.py [gold trees on the test set (same format as sample_input_english.txt)] hyp.trees > hyp_final.trees
    EVALB/evalb -p COLLINS.prm [gold trees on the test set (same format as sample_input_english.txt)] hyp_final.trees > parsing_result.txt

The file `parsing_result.txt` contains the final parsing accuracy using EVALB.

## License
This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Changes

Several major changes has been added from the [original implementation by Dyer et al.](https://github.com/clab/rnng/):

- Implementing pretrained word embeddings feature to the generative model
- Implementing RNNGs with character embeddings
