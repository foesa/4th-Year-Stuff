<template>
  <div class="container">
    <h1> Movies</h1>
    <div class="create-post">
      <input type="text" id="create-post" v-model="year" placeholder="Enter Year">
      <input type="text" id="create-post" v-model="title" placeholder="Enter Title">
      <button v-on:click="getMovies">Get Movies!</button>
      <button v-on:click="createTables">Create Tables!</button>
      <button v-on:click="deleteTables">Delete Tables!</button>
    </div>
    <hr>
    <p class="error" v-if="error">{{error}}</p>
    <div class="posts-container">
      <div class ="posts">
      <table class="center">
        <tr>
          <th>Year </th>
          <th> </th>
          <th>Title</th>
        </tr>
        <tbody>
          <tr v-for="movie in movies" :key="movie._id">
            <td>{{movie.year}}</td>
            <td> </td>
            <td>{{movie.title}}</td>
          </tr>
        </tbody>
      </table>
      </div>
    </div>
  </div>
</template>

<script>
const post_url = 'http://localhost:5000/api/data/';
const delete_url = 'http://localhost:5000/api/data/delete/';
const get_url = 'http://localhost:5000/api/data/'
const axios = require('axios');
export default {
  name: 'FrontEnder',
  data() {
    return{
      movies: [],
      error: "",
      year: "",
      title: "",
    }
  },
  methods: {
    async getMovies(){
      console.log(this.year, this.title);
      await axios.post(post_url , {
            "year":this.year,
            "title":this.title
            }).then((response) => {
            this.movies = response.data;
            console.log(response.data);
             }, (error) => {
            console.log(error);
             });
  },
  async createTables(){
    await axios.get(get_url);
  },
  async deleteTables(){
    await axios.get(delete_url);
  }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
div.container {
  max-width: 800px;
  margin: 0 auto;
}

p.error {
  border: 1px solid #ff5b5f;
  background-color: #ffc5c1;
  padding:  10px;
  margin-bottom: 15px;
}

div.post {
  position: relative;
  border: 1px solid #5bd658;
  background-color:  #bcffb8;
  padding: 10px 10px 30px 10px;
  margin-bottom: 15px;
}

div.created-at {
  position: absolute;
  top: 0;
  left: 0;
  padding: 5px 15px 5px 15px;
  background-color: darkgreen;
  color: white;
  font-size: 13px;
}

p.text {
  font-size: 22px;
  font-weight: 700;
  margin-bottom: 0;
}

.center {
  margin-left: auto;
  margin-right: auto;
}
</style>