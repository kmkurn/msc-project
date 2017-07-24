#include <vector>

#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/model.h"

#include "nt-parser/char-embeddings.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;
using namespace parser::char_embs;

BaseModel::BaseModel(Model *model, unsigned vocab_size, unsigned dim) :
  char_embs(model->add_lookup_parameters(vocab_size, {dim})),
  vocab_size(vocab_size), dim(dim) {}
BaseModel::~BaseModel() { delete char_embs; }

unsigned BaseModel::get_vocab_size() const { return vocab_size; }
unsigned BaseModel::get_dim() const { return dim; }
LookupParameters *BaseModel::get_char_embs() const { return char_embs; }


AdditionModel::AdditionModel(Model *model, unsigned vocab_size, unsigned dim) :
  BaseModel(model, vocab_size, dim) {}

Expression AdditionModel::compute_word_embedding(ComputationGraph *cg, word_t word) {
  vector<Expression> embeddings;
  for (char_t c : word)
    embeddings.push_back(lookup(*cg, char_embs, c));
  return sum(embeddings);
}
