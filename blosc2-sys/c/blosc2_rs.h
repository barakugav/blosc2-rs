#ifndef BLOSC2_RS_H
#define BLOSC2_RS_H

#include "blosc2.h"

#ifdef __cplusplus
extern "C"
{
#endif

  enum blosc2_rs_mmap_mode
  {
    // Open existing file for reading only.
    BLOSC2_RS_MMAP_READ,
    // Open existing file for reading and writing.
    BLOSC2_RS_MMAP_READ_WRITE,
    // Copy-on-write: assignments affect data in memory, but changes are not saved to disk. The file on disk is read-only.
    BLOSC2_RS_MMAP_COW,
  };

  blosc2_schunk *blosc2_rs_schunk_open_mmap(const char *urlpath, int64_t offset, enum blosc2_rs_mmap_mode mode);

#ifdef __cplusplus
}
#endif

#endif /* BLOSC2_RS_H */
