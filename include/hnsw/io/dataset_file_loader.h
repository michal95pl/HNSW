//
// Created by michal on 10.11.2026.
//

#ifndef HNSW_DATASET_LOADER_H
#define HNSW_DATASET_LOADER_H

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

class DatasetFileLoader {
public:
    virtual ~DatasetFileLoader() = default;

    DatasetFileLoader(const DatasetFileLoader&) = delete;
    DatasetFileLoader& operator=(const DatasetFileLoader&) = delete;

    [[nodiscard]] uint64_t getNumRows() const { return numRows; }
    [[nodiscard]] uint64_t getCurrentRow() const { return currentRow; }
    [[nodiscard]] uint64_t getRemainingRows() const { return numRows - currentRow; }
    [[nodiscard]] bool hasNext() const { return currentRow < numRows; }

    bool seekToRow(uint64_t index);

    bool reset() { return seekToRow(0); }

protected:
    DatasetFileLoader(const std::string& path, std::string_view expectedMagic, uint32_t headerSizeBytes);

    /// Reads one header field (directly after the magic number / previous field).
    template <typename T>
        requires std::is_trivially_copyable_v<T>
    bool readHeaderValue(T& out) {
        file.read(reinterpret_cast<char*>(&out), sizeof(T));
        if (!file) {
            return false;
        }
        return true;
    }

    /// Sets the size of a single row, computes number of rows and moves the cursor to the first row.
    void setRowSize(uint64_t bytes);

    /**
     * Reads up to maxRows rows starting at the cursor and moves the cursor.
     * @param dst buffer with room for at least maxRows * getRowSizeBytes() bytes
     * @param maxRows
     * @return number of rows actually read
     */
    uint64_t readRows(char* dst, uint64_t maxRows);

    [[nodiscard]] uint64_t getRowSizeBytes() const { return rowSizeBytes; }

    /**
     * Reads the next n rows.
     * Rows are read in chunks, so the temporary buffer stays small even for huge n.
     * @return false if no rows were read
     */
    template <typename T, typename Decoder>
    bool readNextDecoded(uint64_t n, std::vector<T>& out, Decoder decode) {
        const uint64_t count = std::min(n, getRemainingRows());
        out.resize(count);

        // calculates rows by chunk
        const uint64_t chunkRows = std::max<uint64_t>(1, CHUNK_BYTES / std::max<uint64_t>(1, rowSizeBytes));

        uint64_t done = 0;
        while (done < count) {
            const uint64_t chunk = std::min(count - done, chunkRows);
            rowBuffer.resize(chunk * rowSizeBytes);

            const uint64_t rowsRead = readRows(rowBuffer.data(), chunk);
            for (uint64_t i = 0; i < rowsRead; ++i) {
                decode(rowBuffer.data() + i * rowSizeBytes, out[done + i]);
            }
            done += rowsRead;
            if (rowsRead != chunk) break;
        }

        out.resize(done);
        return done > 0;
    }

    /// Reads all rows from the file. The cursor position is preserved.
    template <typename T, typename Decoder>
    std::vector<T> readAllDecoded(Decoder decode) {
        std::vector<T> all;
        const uint64_t savedRow = currentRow;
        if (reset()) {
            readNextDecoded(numRows, all, decode);
        }
        seekToRow(savedRow);
        return all;
    }

private:
    // 4mb
    static constexpr uint64_t CHUNK_BYTES = 4 * 1024 * 1024;

    std::ifstream file;
    std::string path;
    std::vector<char> rowBuffer;
    uint64_t fileSize = 0;
    uint64_t headerSizeBytes = 0;
    uint64_t rowSizeBytes = 0;
    uint64_t numRows = 0;
    uint64_t currentRow = 0;
};

#endif
