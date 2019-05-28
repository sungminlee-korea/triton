#ifndef TRITON_INCLUDE_AST_OPS_H
#define TRITON_INCLUDE_AST_OPS_H

#include "parser.hpp"
#include <cassert>
#include <vector>
#include <string>
#include <iostream>

namespace triton{
namespace ast{

enum ASSIGN_OP_T{
  ASSIGN,
  INPLACE_MUL, INPLACE_DIV, INPLACE_MOD,
  INPLACE_ADD, INPLACE_SUB,
  INPLACE_LSHIFT, INPLACE_RSHIFT,
  INPLACE_AND, INPLACE_XOR,
  INPLACE_OR
};

enum BIN_OP_T{
  MUL, DIV, MOD,
  ADD, SUB,
  LEFT_SHIFT, RIGHT_SHIFT,
  LT, GT,
  LE, GE,
  EQ, NE,
  AND, XOR, OR,
  LAND, LOR
};

enum UNARY_OP_T{
  INC, DEC,
  PLUS, MINUS,
  ADDR, DEREF,
  COMPL, NOT
};

enum TYPE_T{
  VOID_T,
  UINT1_T, UINT8_T, UINT16_T, UINT32_T, UINT64_T,
  INT1_T, INT8_T, INT16_T, INT32_T, INT64_T,
  FLOAT32_T, FLOAT64_T
};

enum STORAGE_SPEC_T{
  CONST_T,
  TUNABLE_T,
  KERNEL_T,
  RESTRICT_T,
  READONLY_T,
  CONSTANT_SPACE_T,
  WRITEONLY_T
};

}
}

#endif
