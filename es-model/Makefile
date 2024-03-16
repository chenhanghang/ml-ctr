CC = g++
C11 = -std=c++11 -g


# path assignment
WORK_DIR = ./
INC_PATH = $(join $(WORK_DIR), include)
SRC_PATH = $(join $(WORK_DIR), src)
OBJ_PATH = $(join $(WORK_DIR), obj)
BIN_PATH = $(join $(WORK_DIR), bin)

# src/*.cpp wildcard 获取指定文件
SRC_FILE_WITH_DIR = $(wildcard $(join $(SRC_PATH), /*.cpp))
# *.cpp
SRC_FILE = $(notdir $(SRC_FILE_WITH_DIR))

# *.o
OBJ_FILE = $(SRC_FILE:%.cpp=%.o)
# obj/*.o
OBJ_FILE_WITH_DIR = $(patsubst %.o, $(OBJ_PATH)/%.o, $(OBJ_FILE))

TARGET = $(join $(BIN_PATH), /es_model)

$(OBJ_PATH)/%.o : $(SRC_PATH)/%.cpp
	$(CC) $(C11) -I$(INC_PATH) -c $< -o $@

$(TARGET) : $(OBJ_FILE_WITH_DIR)
	$(CC) $(C11) -lgflags -o $@ $^

.PHONY: clean
clean:
	@rm $(OBJ_FILE_WITH_DIR) # @表示不在终端打印执行的命令
	@rm $(TARGET)

