#0 building with "desktop-linux" instance using docker driver

#1 [internal] load build definition from Dockerfile
#1 transferring dockerfile: 344B 0.0s done
#1 DONE 0.1s

#2 [internal] load metadata for docker.io/library/python:3.11-slim
#2 ...

#3 [auth] library/python:pull token for registry-1.docker.io
#3 DONE 0.0s

#2 [internal] load metadata for docker.io/library/python:3.11-slim
#2 DONE 1.0s

#4 [internal] load .dockerignore
#4 transferring context: 2B done
#4 DONE 0.0s

#5 [1/3] FROM docker.io/library/python:3.11-slim@sha256:158caf0e080e2cd74ef2879ed3c4e697792ee65251c8208b7afb56683c32ea6c
#5 resolve docker.io/library/python:3.11-slim@sha256:158caf0e080e2cd74ef2879ed3c4e697792ee65251c8208b7afb56683c32ea6c 0.0s done
#5 DONE 0.2s

#5 [1/3] FROM docker.io/library/python:3.11-slim@sha256:158caf0e080e2cd74ef2879ed3c4e697792ee65251c8208b7afb56683c32ea6c
#5 sha256:3f0cdbca744e7bd0ce0ff6da73b9148829b04309925992954a314ba203f56e99 249B / 249B 0.1s done
#5 sha256:4d55cfecf3663813d03c369bcd532b89f41cf07b65d95887ef686538370a747c 0B / 14.36MB 0.3s
#5 sha256:72cf4c3b83019e176aba979aba419d35f56576bbcfc4f7249a1ab1d4b536730b 0B / 1.29MB 0.1s
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 0B / 29.78MB 0.1s
#5 sha256:4d55cfecf3663813d03c369bcd532b89f41cf07b65d95887ef686538370a747c 2.10MB / 14.36MB 0.4s
#5 sha256:4d55cfecf3663813d03c369bcd532b89f41cf07b65d95887ef686538370a747c 4.19MB / 14.36MB 0.6s
#5 sha256:72cf4c3b83019e176aba979aba419d35f56576bbcfc4f7249a1ab1d4b536730b 1.29MB / 1.29MB 0.5s done
#5 sha256:4d55cfecf3663813d03c369bcd532b89f41cf07b65d95887ef686538370a747c 6.29MB / 14.36MB 0.7s
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 2.10MB / 29.78MB 0.6s
#5 sha256:4d55cfecf3663813d03c369bcd532b89f41cf07b65d95887ef686538370a747c 8.39MB / 14.36MB 0.9s
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 4.19MB / 29.78MB 0.7s
#5 sha256:4d55cfecf3663813d03c369bcd532b89f41cf07b65d95887ef686538370a747c 10.49MB / 14.36MB 1.0s
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 6.29MB / 29.78MB 0.9s
#5 sha256:4d55cfecf3663813d03c369bcd532b89f41cf07b65d95887ef686538370a747c 12.58MB / 14.36MB 1.2s
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 8.39MB / 29.78MB 1.0s
#5 sha256:4d55cfecf3663813d03c369bcd532b89f41cf07b65d95887ef686538370a747c 14.36MB / 14.36MB 1.3s done
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 12.58MB / 29.78MB 1.3s
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 14.68MB / 29.78MB 1.5s
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 16.78MB / 29.78MB 1.6s
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 19.92MB / 29.78MB 1.8s
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 22.02MB / 29.78MB 1.9s
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 25.17MB / 29.78MB 2.1s
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 27.26MB / 29.78MB 2.2s
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 29.78MB / 29.78MB 2.4s
#5 sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 29.78MB / 29.78MB 2.4s done
#5 extracting sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5
#5 extracting sha256:1733a4cd59540b3470ff7a90963bcdea5b543279dd6bdaf022d7883fdad221e5 0.4s done
#5 DONE 3.0s

#5 [1/3] FROM docker.io/library/python:3.11-slim@sha256:158caf0e080e2cd74ef2879ed3c4e697792ee65251c8208b7afb56683c32ea6c
#5 extracting sha256:72cf4c3b83019e176aba979aba419d35f56576bbcfc4f7249a1ab1d4b536730b 0.1s done
#5 extracting sha256:4d55cfecf3663813d03c369bcd532b89f41cf07b65d95887ef686538370a747c
#5 extracting sha256:4d55cfecf3663813d03c369bcd532b89f41cf07b65d95887ef686538370a747c 0.2s done
#5 DONE 3.3s

#5 [1/3] FROM docker.io/library/python:3.11-slim@sha256:158caf0e080e2cd74ef2879ed3c4e697792ee65251c8208b7afb56683c32ea6c
#5 extracting sha256:3f0cdbca744e7bd0ce0ff6da73b9148829b04309925992954a314ba203f56e99 0.0s done
#5 DONE 3.4s

#6 [2/3] RUN useradd -m -s /bin/bash tars
#6 DONE 0.5s

#7 [3/3] WORKDIR /home/tars/workspace
#7 DONE 0.1s

#8 exporting to image
#8 exporting layers
#8 exporting layers 0.2s done
#8 exporting manifest sha256:b8480df8289918575d803b67dd7d09285958a816c8fbf5215bd027fdf907a848 0.0s done
#8 exporting config sha256:b92fa16e1385d159d4deb0165e7ce0c91492fe08ac20fee79323570bc15bc049 0.0s done
#8 exporting attestation manifest sha256:c25451be32b7c75a9c018af69e8f005ca708efbdf25a39ded19c3ff704d5e5ae 0.0s done
#8 exporting manifest list sha256:a27fcc490c919dadeadde5aa47ff04c162642a4e8e0d629d3bd191a571056b10 0.0s done
#8 naming to docker.io/library/tars-sandbox:latest
#8 naming to docker.io/library/tars-sandbox:latest done
#8 unpacking to docker.io/library/tars-sandbox:latest 0.1s done
#8 DONE 0.4s

View build details: docker-desktop://dashboard/build/desktop-linux/desktop-linux/ko79e9oy6vnvw4f3mezdfdife
