#ifndef PARSER_CHAR_EMBEDDINGS_H_
#define PARSER_CHAR_EMBEDDINGS_H_

#include <vector>

#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/model.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

namespace parser {
  namespace char_embs {
    using char_t = int;
    using word_t = vector<char_t>;

    /**
       Abstract base class for char embeddings model.
    */
    class BaseModel {
    public:
      unsigned get_vocab_size() const;
      unsigned get_dim() const;
      LookupParameters* get_char_embs() const;
      virtual Expression compute_word_embedding(ComputationGraph *cg, word_t word) = 0;
    protected:
      LookupParameters *char_embs;
      unsigned vocab_size;
      unsigned dim;

      BaseModel(Model *model, unsigned vocab_size, unsigned dim);
      virtual ~BaseModel();
    };

    class AdditionModel : public BaseModel {
    public:
      AdditionModel(Model *model, unsigned vocab_size, unsigned dim);
      virtual Expression compute_word_embedding(ComputationGraph *cg, word_t word) override;
    };
  }
}

#endif
