CC = g++
C11 = -Wall -std=c++11 -g 


# path assignment
WORK_DIR = ./
SRC_PATH = $(join $(WORK_DIR), src)
OBJ_PATH = $(join $(WORK_DIR), obj)
BIN_PATH = $(join $(WORK_DIR), bin)
MODEL_PATH = $(join $(WORK_DIR), model)


DIRS	=$(shell find ./src -maxdepth 3 -type d)
SOURCE	= $(foreach dir,$(DIRS),$(wildcard $(dir)/*.cpp))
OBJS    = $(patsubst %.cpp,%.o,$(patsubst %.cpp,%.o,$(SOURCE)))


# src/*.cpp wildcard 获取指定文件
#SRC_FILE_WITH_DIR = $(wildcard $(join $(SRC_PATH), /*.cpp))
SRC_FILE_WITH_DIR = $(foreach dir,$(DIRS),$(wildcard $(dir)/*.cpp))
# *.cpp
SRC_FILE = $(notdir $(SRC_FILE_WITH_DIR))

# *.o
OBJ_FILE = $(SRC_FILE:%.cpp=%.o)
# obj/*.o
OBJ_FILE_WITH_DIR = $(patsubst %.o, $(OBJ_PATH)/%.o, $(OBJ_FILE))

TARGET = $(join $(BIN_PATH), /ctr_predict)


#$(OBJS) : $(SRC_FILE_WITH_DIR)
#	$(CC) $(C11) -I. -c $< -o $@

$(OBJ_PATH)/%.o : $(SRC_PATH)/%.cpp
	$(CC) $(C11) -I. -c $< -o $@
$(OBJ_PATH)/%.o : $(SRC_PATH)/frame/%.cpp
	$(CC) $(C11) -I. -c $< -o $@
$(OBJ_PATH)/%.o : $(SRC_PATH)/sample/%.cpp
	$(CC) $(C11) -I. -c $< -o $@
$(OBJ_PATH)/%.o : $(SRC_PATH)/utils/%.cpp
	$(CC) $(C11) -I. -c $< -o $@
$(OBJ_PATH)/%.o : $(SRC_PATH)/models/%.cpp
	$(CC) $(C11) -I. -c $< -o $@
$(OBJ_PATH)/%.o : $(SRC_PATH)/models/lr/%.cpp
	$(CC) $(C11) -I. -c $< -o $@
$(OBJ_PATH)/%.o : $(SRC_PATH)/models/fm/%.cpp
	$(CC) $(C11) -I. -c $< -o $@
$(OBJ_PATH)/%.o : $(SRC_PATH)/models/fm_ftrl/%.cpp
	$(CC) $(C11) -I. -c $< -o $@
$(OBJ_PATH)/%.o : $(SRC_PATH)/models/ffm/%.cpp
	$(CC) $(C11) -I. -c $< -o $@
$(OBJ_PATH)/%.o : $(SRC_PATH)/models/fm_pairwise/%.cpp
	$(CC) $(C11) -I. -c $< -o $@

$(TARGET) : $(OBJ_FILE_WITH_DIR)
	$(CC) -O3 $(C11) -lgflags -lpthread -o $@ $^


.PHONY: clean
clean:
	@echo $(OBJS)
	@rm $(OBJ_FILE_WITH_DIR) # @表示不在终端打印执行的命令
	@echo $(TARGET)
	@rm $(TARGET)
	@rm $(MODEL_PATH)/* 
