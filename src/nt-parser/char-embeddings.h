#ifndef PARSER_CHAR_EMBEDDINGS_H_
#define PARSER_CHAR_EMBEDDINGS_H_

#include <vector>

#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/model.h"

namespace parser {
  namespace char_embs {
    using char_t = int;
    using word_t = std::vector<char_t>;

    /**
       Abstract base class for char embeddings model.
    */
    class BaseModel {
    public:
      unsigned get_vocab_size() const;
      unsigned get_dim() const;
      cnn::LookupParameters *get_char_embs() const;
      virtual cnn::expr::Expression compute_word_embedding(cnn::ComputationGraph *cg,
                                                           word_t word) = 0;
    protected:
      cnn::LookupParameters *char_embs;
      unsigned vocab_size;
      unsigned dim;

      BaseModel(cnn::Model *model, unsigned vocab_size, unsigned dim);
      virtual ~BaseModel();
    };

    class AdditionModel : public BaseModel {
    public:
      AdditionModel(cnn::Model *model, unsigned vocab_size, unsigned dim);
      virtual cnn::expr::Expression compute_word_embedding(cnn::ComputationGraph *cg,
                                                           word_t word) override;
    };
  }
}

#endif
