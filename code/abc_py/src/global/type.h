/**
 * @file type.h
 * @brief Define some the types being used globally
 * @author Keren Zhu
 * @date 09/30/2019
 */

#ifndef ABC_PY_TYPE_H_
#define ABC_PY_TYPE_H_

#include <cstdint>
#include <string>
#include <sstream>
#include "namespace.h"

PROJECT_NAMESPACE_BEGIN
// Built-in type aliases
using IndexType  = std::uint32_t;
using IntType    = std::int32_t;
using RealType   = double;
using Byte       = std::uint8_t;
using LocType    = std::int32_t; // Location/design unit // Location/design unit
 // Location/design unit
// Built-in type constants
constexpr IndexType INDEX_TYPE_MAX  = UINT32_MAX;
constexpr IntType INT_TYPE_MAX      = INT32_MAX;
constexpr IntType INT_TYPE_MIN      = INT32_MIN;
constexpr RealType REAL_TYPE_MAX    = 1e100;
constexpr RealType REAL_TYPE_MIN    = -1e100;
constexpr RealType REAL_TYPE_TOL    = 1e-6;
constexpr LocType LOC_TYPE_MAX      = INT32_MAX;
constexpr LocType LOC_TYPE_MIN      = INT32_MIN;

// Type aliases
//using CostTy     = double;
using CostType = RealType;
constexpr CostType COST_TYPE_INVALID = REAL_TYPE_MIN;
constexpr  CostType COST_TYPE_MAX = REAL_TYPE_MAX;


// Enums




PROJECT_NAMESPACE_END

#endif // ABC_PY_TYPE_H_

