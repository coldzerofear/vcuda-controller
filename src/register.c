/*
 * Tencent is pleased to support the open source community by making TKEStack available.
 *
 * Copyright (C) 2012-2019 Tencent. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use
 * this file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OF ANY KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations under the License.
 */

//
// Created by Thomas Song on 2019-04-15.
//
#include <errno.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "include/hijack.h"

static const struct timespec g_cycle = {
    .tv_sec = 0,
    .tv_nsec = TIME_TICK * MILLISEC,
};

// #lizard forgives
void register_to_remote_with_data(const char* bus_id, const char* pod_uid,
                                  const char* container, const char* cgroup_path) {
  pid_t register_pid;
  int wstatus = 0, wret = 0;
  pid_t child_pid;
  int pipe_fd[2];
  int ret = -1;
  // 创建一个管道，以实现进程间的通信
  ret = pipe(pipe_fd);
  if (unlikely(ret)) {
    // 少时情况下可能出现管道创建错误，打印日志
    LOGGER(FATAL, "create pipe failed, error %s", strerror(errno));
  }
  //系统函数，用来创建新进程
  register_pid = fork();
  if (!register_pid) { // 子进程创建成功
    // 关闭父进程id
    close(pipe_fd[1]);
    // 复制子进程id
    while (read(pipe_fd[0], &child_pid, sizeof(pid_t)) == 0) {
      nanosleep(&g_cycle, NULL);
    }
    // 子进程执行下面的操作 
    // child
    if (is_custom_config_path()) {
      ret = execl((RPC_CLIENT_PATH RPC_CLIENT_NAME), RPC_CLIENT_NAME, "--addr",
                  RPC_ADDR, "--bus-id", bus_id, "--pod-uid", pod_uid,
                  "--cont-id", container, "--cgroup-path", cgroup_path, (char*)NULL);
    } else {
      ret = execl((RPC_CLIENT_PATH RPC_CLIENT_NAME), RPC_CLIENT_NAME, "--addr",
                  RPC_ADDR, "--bus-id", bus_id, "--pod-uid", pod_uid,
                  "--cont-name", container, "--cgroup-path", cgroup_path, (char*)NULL);
    }
    if (unlikely(ret == -1)) {
      LOGGER(FATAL, "can't register to manager, error %s", strerror(errno));
    }

    // 关闭子进程id
    close(pipe_fd[0]);
    // 退出返回0：执行成功
    _exit(EXIT_SUCCESS);
  } else { // 子进程创建失败
    // 关闭子进程id
    close(pipe_fd[0]);
    
    while (write(pipe_fd[1], &register_pid, sizeof(pid_t)) == 0) {
      nanosleep(&g_cycle, NULL);
    }

    do {
      wret = waitpid(register_pid, &wstatus, WUNTRACED | WCONTINUED);
      if (unlikely(wret == -1)) {
        LOGGER(FATAL, "waitpid failed, error %s", strerror(errno));
      }
    } while (!WIFEXITED(wstatus) && !WIFSIGNALED(wstatus));

    ret = WEXITSTATUS(wstatus);
    if (unlikely(ret)) {
      LOGGER(FATAL, "rpc client exit with %d", ret);
    }

    close(pipe_fd[1]);
  }
}
