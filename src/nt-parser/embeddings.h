#ifndef PARSER_EMBEDDINGS_H_
#define PARSER_EMBEDDINGS_H_

#include <string>

#include "cnn/cnn.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/model.h"

namespace parser {
  namespace embeddings {
    class BaseModel {
    public:
      unsigned get_vocab_size() const;
      unsigned get_dim() const;
      cnn::LookupParameters& get_embeddings() const;
      cnn::Dict& get_term_dict() const;
      virtual cnn::expr::Expression compute_word_embedding(cnn::ComputationGraph &cg,
                                                           std::string word) = 0;

    protected:
      cnn::LookupParameters &embeddings;
      unsigned dim;
      cnn::Dict &term_dict;

      BaseModel(cnn::Model &model, cnn::Dict &term_dict, unsigned dim);
    };

    namespace word {
      class SimpleLookupModel : public BaseModel {
      public:
        SimpleLookupModel(cnn::Model &model, cnn::Dict &word_dict, unsigned dim);
        cnn::expr::Expression compute_word_embedding(cnn::ComputationGraph &cg,
                                                     std::string word) override;
      };
    }

    namespace character {
      class AdditionModel : public BaseModel {
      public:
        AdditionModel(cnn::Model &model, cnn::Dict &char_dict, unsigned dim);
        cnn::expr::Expression compute_word_embedding(cnn::ComputationGraph &cg,
                                                     std::string word) override;
      };
    }
  }
}

#endif
