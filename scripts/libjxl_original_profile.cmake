# Inject the public-C-API test oracle without editing the pinned libjxl source.
if(PROJECT_NAME STREQUAL "LIBJXL" AND NOT TARGET jxl-original-profile)
  add_executable(jxl-original-profile "${CMAKE_CURRENT_LIST_DIR}/libjxl_original_profile.cc")
  target_compile_features(jxl-original-profile PRIVATE cxx_std_17)
  target_include_directories(jxl-original-profile PRIVATE
    "${PROJECT_SOURCE_DIR}/lib/include" "${PROJECT_BINARY_DIR}/lib/include")
  target_link_libraries(jxl-original-profile PRIVATE jxl jxl_cms)
  set_target_properties(jxl-original-profile PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tools")
endif()

if(PROJECT_NAME STREQUAL "LIBJXL" AND NOT TARGET jxl-cmyk-frames)
  add_executable(jxl-cmyk-frames "${CMAKE_CURRENT_LIST_DIR}/cmyk_raw_frame_oracle.cc")
  target_compile_features(jxl-cmyk-frames PRIVATE cxx_std_17)
  target_include_directories(jxl-cmyk-frames PRIVATE
    "${PROJECT_SOURCE_DIR}/lib/include" "${PROJECT_BINARY_DIR}/lib/include")
  target_link_libraries(jxl-cmyk-frames PRIVATE jxl jxl_cms)
  set_target_properties(jxl-cmyk-frames PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tools")
endif()
