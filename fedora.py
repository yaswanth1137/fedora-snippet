import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

sample_logs = [
    "Apr 7 21:45:13 fedora sshd[12345]: Failed password for invalid user admin from 192.168.1.10 port 22 ssh2",
    "Apr 7 21:47:22 fedora sshd[12346]: Failed password for root from 192.168.1.11 port 22 ssh2",
    "Apr 7 22:15:42 fedora sudo: pam_unix(sudo:auth): authentication failure; logname=user uid=1000 euid=0 tty=/dev/pts/0 ruser=user rhost= user=user",
    "Apr 7 22:30:19 fedora kernel: [UFW BLOCK] IN=eth0 OUT= MAC=00:11:22:33:44:55 SRC=10.0.0.5 DST=10.0.0.1 LEN=40 TOS=0x00 PREC=0x00 TTL=249 ID=8475 PROTO=TCP SPT=37386 DPT=22 WINDOW=1024 RES=0x00 SYN URGP=0",
    "Apr 7 23:01:12 fedora firewalld[1234]: WARNING: Dropping incoming packet from 192.168.1.15 to 192.168.1.1",
    "Apr 8 00:12:32 fedora sshd[12347]: Invalid user test from 10.0.0.100",
    "Apr 8 00:15:12 fedora sshd[12348]: User root from 192.168.1.20 not allowed because not listed in AllowUsers",
    "Apr 8 01:25:34 fedora su: FAILED SU (to root) user on /dev/pts/0",
    "Apr 8 02:10:23 fedora sudo: user : 3 incorrect password attempts ; TTY=pts/0 ; PWD=/home/user ; USER=root ; COMMAND=/bin/bash",
    "Apr 8 03:48:12 fedora kernel: Possible SYN flooding on port 80. Sending cookies.",
    
    "Apr 7 12:01:15 fedora systemd[1]: Started User Manager for UID 1000.",
    "Apr 7 12:05:22 fedora dbus-daemon[1234]: [system] Successfully activated service 'org.freedesktop.nm_dispatcher'",
    "Apr 7 12:10:32 fedora NetworkManager[1234]: <info>  [1617786632.1234] manager: NetworkManager state is now CONNECTED_GLOBAL",
    "Apr 7 12:15:40 fedora gdm-password][1234]: gkr-pam: unlocked login keyring",
    "Apr 7 12:20:12 fedora gnome-shell[1234]: Window manager warning: Buggy client sent a _NET_ACTIVE_WINDOW message with a timestamp of 0",
    "Apr 7 12:30:25 fedora PackageKit: daemon start",
    "Apr 7 12:35:45 fedora cupsd[1234]: Scheduler shutting down normally.",
    "Apr 7 12:40:10 fedora dhclient[1234]: DHCPREQUEST on enp0s3 to 192.168.1.1 port 67",
    "Apr 7 12:45:20 fedora kernel: [drm:drm_atomic_helper_commit_cleanup_done [drm_kms_helper]] Committed atomic update took 10897 us",
    "Apr 7 12:50:35 fedora systemd[1]: Starting Cleanup of Temporary Directories...",
    "Apr 7 12:55:42 fedora systemd-logind[1234]: New session 3 of user user.",
    "Apr 7 13:00:15 fedora bluetoothd[1234]: Endpoint registered: sender=:1.65 path=/MediaEndpoint/A2DPSource/sbc",
    "Apr 7 13:05:22 fedora pulseaudio[1234]: [pulseaudio] sink-input.c: Failed to create sink input: sink is suspended.",
    "Apr 7 13:10:30 fedora gnome-software[1234]: enabled plugins: desktop-categories, fwupd, os-release, appstream, hardcoded-popular, rewrite-resource, hardcoded-featured, hardcoded-blacklist, desktop-menu-path, flatpak, packagekit-local, packagekit-url, systemd-updates, packagekit, packagekit-refresh, packagekit-offline, packagekit-proxy, repos, generic-updates, provenance, modalias, desktop-categories, hardcoded-popular, desktop-menu-path, rewrite-resource, packagekit, packagekit-proxy, packagekit-refresh, appstream, provenance, repos, packagekit-url, packagekit-local, generic-updates, flatpak, systemd-updates, odrs, os-release, packagekit-offline, fwupd, modalias",
    "Apr 7 13:15:45 fedora dbus-daemon[1234]: [system] Activating via systemd: service name='org.freedesktop.hostname1' unit='dbus-org.freedesktop.hostname1.service' requested by ':1.102' (uid=1000 pid=7812 comm=\"gnome-control-c\" label=\"unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023\")",
    "Apr 7 13:20:50 fedora systemd[1]: Started Process Core Dump (PID 8765/UID 0).",
    "Apr 7 13:25:10 fedora tracker-miner-fs[1234]: Tracker file system miner finished initial crawling",
    "Apr 7 13:30:22 fedora audit[1234]: USER_AUTH pid=12345 uid=0 auid=1000 ses=3 msg='op=PAM:authentication grantors=pam_succeed_if,pam_unix acct=\"user\" exe=\"/usr/bin/sudo\" hostname=? addr=? terminal=/dev/pts/0 res=success'",
    "Apr 7 13:35:37 fedora kernel: [drm:amdgpu_job_timedout [amdgpu]] *ERROR* ring gfx timeout, signaled seq=9412, emitted seq=9414",
    "Apr 7 13:40:48 fedora audit: NETFILTER_CFG table=filter family=2 entries=0"
]

labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def preprocess_log(log):
    log = log.lower()
    log = re.sub(r'^[a-z]+ \d+ \d+:\d+:\d+ \w+ ', '', log)
    log = re.sub(r'\[\d+\]', '', log)
    log = re.sub(r'\d+\.\d+\.\d+\.\d+', 'ip_addr', log)
    log = re.sub(r'\d+', 'num', log)
    return log

def extract_features(logs):
    preprocessed_logs = [preprocess_log(log) for log in logs]
    vectorizer = TfidfVectorizer(
        max_features=100,
        min_df=2,
        stop_words='english',
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(preprocessed_logs)
    return X, vectorizer

def main():
    X, vectorizer = extract_features(sample_logs)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42
    )
    
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Security Alert']))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    def classify_log(log_text):
        preprocessed = preprocess_log(log_text)
        features = vectorizer.transform([preprocessed])
        prediction = clf.predict(features)[0]
        probability = clf.predict_proba(features)[0]
        return "Security Alert" if prediction == 1 else "Normal", probability
    
    print("\nClassifying new logs:")
    test_logs = [
        "Apr 8 08:30:12 fedora sshd[12345]: Failed password for root from 192.168.1.100 port 22 ssh2",
        "Apr 8 08:35:45 fedora systemd[1]: Started GNOME Display Manager."
    ]
    
    for log in test_logs:
        classification, proba = classify_log(log)
        print(f"\nLog: {log}")
        print(f"Classification: {classification}")
        if classification == "Security Alert":
            print(f"Confidence: {proba[1]:.2f}")
        else:
            print(f"Confidence: {proba[0]:.2f}")
            
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(clf, "feature_log_prob_"):
        security_alert_features = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        top_security_indices = np.argsort(security_alert_features)[-10:]
        print("\nTop features for security alerts:")
        for idx in top_security_indices:
            print(f"- {feature_names[idx]}")

if __name__ == "__main__":
    main()
