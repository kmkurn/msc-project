#include <string>
#include <vector>

#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/model.h"

#include "nt-parser/embeddings.h"

using namespace cnn;
using namespace cnn::expr;

using std::string;
using std::vector;

namespace emb = parser::embeddings;
namespace chr = parser::embeddings::character;

emb::BaseModel::BaseModel(Model &model, Dict &term_dict, unsigned dim) :
  embeddings(*model.add_lookup_parameters(term_dict.size(), {dim})),
  dim(dim), term_dict(term_dict) {}

unsigned emb::BaseModel::get_vocab_size() const { return term_dict.size(); }
unsigned emb::BaseModel::get_dim() const { return dim; }
LookupParameters& emb::BaseModel::get_embeddings() const { return embeddings; }
Dict& emb::BaseModel::get_term_dict() const { return term_dict; }


chr::AdditionModel::AdditionModel(Model &model, Dict& char_dict, unsigned dim) :
  BaseModel(model, char_dict, dim) {}

Expression chr::AdditionModel::compute_word_embedding(ComputationGraph &cg, string word) {
  vector<Expression> temp;
  for (char c : word) {
    string s(1, c);
    temp.push_back(lookup(cg, &embeddings, term_dict.Convert(s)));
  }
  return sum(temp);
}
