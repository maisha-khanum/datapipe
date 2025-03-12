# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_smartbelt_segmentation_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED smartbelt_segmentation_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(smartbelt_segmentation_FOUND FALSE)
  elseif(NOT smartbelt_segmentation_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(smartbelt_segmentation_FOUND FALSE)
  endif()
  return()
endif()
set(_smartbelt_segmentation_CONFIG_INCLUDED TRUE)

# output package information
if(NOT smartbelt_segmentation_FIND_QUIETLY)
  message(STATUS "Found smartbelt_segmentation: 0.0.0 (${smartbelt_segmentation_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'smartbelt_segmentation' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${smartbelt_segmentation_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(smartbelt_segmentation_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${smartbelt_segmentation_DIR}/${_extra}")
endforeach()
