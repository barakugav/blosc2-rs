#include "blosc2_rs.h"

blosc2_schunk *blosc2_rs_schunk_open_mmap(const char *urlpath, int64_t offset, enum blosc2_rs_mmap_mode mode)
{
  const char *mode_str;
  switch (mode)
  {
  case BLOSC2_RS_MMAP_READ:
    mode_str = "r";
    break;
  case BLOSC2_RS_MMAP_READ_WRITE:
    mode_str = "r+";
    break;
  case BLOSC2_RS_MMAP_COW:
    mode_str = "c";
    break;
  default:
    BLOSC_TRACE_ERROR("Invalid mmap_mode");
    return NULL;
  }

  blosc2_stdio_mmap *mmap_file = malloc(sizeof(blosc2_stdio_mmap));
  *mmap_file = BLOSC2_STDIO_MMAP_DEFAULTS;
  mmap_file->needs_free = true;
  mmap_file->mode = mode_str;
  blosc2_io io = {.id = BLOSC2_IO_FILESYSTEM_MMAP, .params = mmap_file};
  blosc2_schunk *schunk = blosc2_schunk_open_offset_udio(urlpath, offset, &io);
  if (schunk == NULL)
  {
    free(mmap_file);
    return NULL;
  }

  if (!schunk->storage->contiguous)
  {
    BLOSC_TRACE_ERROR("Can't open sparse frame with mmap");
    blosc2_schunk_free(schunk);
    schunk = NULL;
  }
  return schunk;
}
