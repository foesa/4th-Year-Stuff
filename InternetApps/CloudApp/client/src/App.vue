<template>
  <div id="app">
    <img alt="Vue logo" src="./assets/logo.png">
    <FrontEnder />
  </div>
</template>

<script>
import FrontEnder from './components/FrontEnder.vue'

export default {
  name: 'App',
  components: {
    FrontEnder
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
