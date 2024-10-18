add_triton_library(TritonSharedAnalysisStructured
  PtrAnalysis.cpp
  OpFoldResultUtils.cpp
  MaskAnalysis.cpp

  DEPENDS
  TritonAnalysis
  TritonTableGen
  TritonStructuredTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonStructuredIR
  MLIRAnalysis
)
