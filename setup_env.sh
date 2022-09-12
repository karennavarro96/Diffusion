# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/n/sw/eb/apps/centos7/Anaconda3/2020.11/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh" ]; then
        . "/n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh"
    else
        export PATH="/n/sw/eb/apps/centos7/Anaconda3/2020.11/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

module load Anaconda3